"""Filter Typhoon dataset and convert to Apple Foundation Model JSONL format.

Apple expects each JSONL line to be:
  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

Conversion rules:
  - Drop system messages (prepend to first user message if Thai-relevant)
  - Drop rows with tool_calls
  - Ensure alternating user/assistant pattern
  - Filter: must contain Thai, Thai ratio > 5%, estimated tokens < 3500
  - Deduplicate by hashing first user message
  - 80/20 train/eval split
"""

import hashlib
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

SEED = 42
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "iteration_1"

THAI_RE = re.compile(r"[\u0E00-\u0E7F]")

# --- Target dataset composition ---
TARGET_SIZE = 4000
SOURCE_TARGETS = {
    "autoif": 0.60,          # Tier 1: Thai alignment (was "typhoon_autoif" in plan)
    "sovereign": 0.20,       # Tier 1b: Thai domain/legal
    "tulu_sft": 0.20,        # Tier 2: bilingual — ai2-adapt-dev/* and other Tulu sources
}


def contains_thai(text: str) -> bool:
    return bool(THAI_RE.search(text))


def thai_ratio(text: str) -> float:
    if not text:
        return 0.0
    thai_chars = len(THAI_RE.findall(text))
    total = len(text.strip())
    return thai_chars / total if total else 0.0


def estimate_tokens(text: str) -> int:
    """Proxy token count. Thai ~2.5 chars/token, Latin ~4 chars/token."""
    thai_chars = len(THAI_RE.findall(text))
    other_chars = len(text) - thai_chars
    return int(thai_chars / 2.5 + other_chars / 4.0)


def conversation_text(messages) -> str:
    """Extract text from messages, handling JSON strings, dicts, and string elements."""
    if isinstance(messages, str):
        # Try parsing as JSON first
        try:
            parsed = json.loads(messages)
            if isinstance(parsed, list):
                return conversation_text(parsed)
        except (json.JSONDecodeError, TypeError):
            return messages
    if not isinstance(messages, list):
        return ""
    parts = []
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content", "")
            if isinstance(c, str):
                parts.append(c)
        elif isinstance(m, str):
            parts.append(m)
    return " ".join(parts)


def parse_messages(raw) -> list[dict] | None:
    """Parse messages from various formats: JSON string, list of dicts, list of strings."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(raw, list):
        return raw
    return None


def normalize_messages(raw_messages) -> list[dict] | None:
    """Convert Typhoon OpenAI chat format to Apple format.

    Returns None if the conversation can't be normalized.
    Handles messages as JSON strings, list of dicts, or list of strings.
    """
    messages = parse_messages(raw_messages)
    if not messages:
        return None

    # Skip rows where messages are strings (not structured chat)
    if messages and isinstance(messages[0], str):
        return None

    system_content = None
    normalized: list[dict] = []

    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role", "")
        content = msg.get("content", "")

        if not isinstance(content, str) or not content.strip():
            # Skip empty or non-string content (e.g. tool_calls)
            if role == "assistant" and msg.get("tool_calls"):
                return None  # Skip tool-call conversations entirely
            continue

        if role == "system":
            if not normalized:
                system_content = content
            # Skip mid-conversation system messages
            continue

        if role == "user":
            # Prepend system content to first user message
            if system_content and not normalized:
                content = f"{system_content}\n\n{content}"
                system_content = None
            normalized.append({"role": "user", "content": content})

        elif role == "assistant":
            normalized.append({"role": "assistant", "content": content})

    if not normalized or len(normalized) < 2:
        return None

    # Must start with user, end with assistant
    if normalized[0]["role"] != "user" or normalized[-1]["role"] != "assistant":
        return None

    # Verify alternating pattern
    for i in range(len(normalized) - 1):
        if normalized[i]["role"] == normalized[i + 1]["role"]:
            return None

    return normalized


def is_valid(messages: list[dict], max_tokens: int = 3500) -> tuple[bool, str]:
    """Validate a normalized message list."""
    full_text = conversation_text(messages)

    if not contains_thai(full_text):
        return False, "no_thai"

    if thai_ratio(full_text) < 0.05:
        return False, "thai_ratio_low"

    if estimate_tokens(full_text) > max_tokens:
        return False, "too_long"

    # Check individual messages aren't empty
    for msg in messages:
        if len(msg["content"].strip()) < 5:
            return False, "empty_message"

    return True, "ok"


def process_source(
    rows: list[dict],
    source_name: str,
    target_count: int,
    require_thai_filter: bool = False,
) -> list[dict]:
    """Process rows from a single source, returning converted samples."""
    collected = []
    stats: dict[str, int] = Counter()

    random.shuffle(rows)

    for row in tqdm(rows, desc=source_name, leave=False):
        if len(collected) >= target_count:
            break

        messages = row.get("messages", [])

        # For tulu_3_sft: pre-filter to only Thai-containing rows
        if require_thai_filter:
            full_text = conversation_text(messages)
            if not contains_thai(full_text):
                stats["skipped_no_thai"] += 1
                continue

        normalized = normalize_messages(messages)
        if normalized is None:
            stats["normalize_failed"] += 1
            continue

        valid, reason = is_valid(normalized)
        if not valid:
            stats[f"invalid_{reason}"] += 1
            continue

        collected.append({"messages": normalized, "source": source_name})
        stats["accepted"] += 1

    console.print(f"  {source_name}: collected {len(collected):,} / target {target_count:,}")
    for k, v in sorted(stats.items()):
        console.print(f"    {k}: {v:,}")

    return collected


def deduplicate(samples: list[dict]) -> list[dict]:
    """Deduplicate by hashing first user message."""
    seen = set()
    unique = []
    for sample in samples:
        first_user = next(
            (m["content"] for m in sample["messages"] if m["role"] == "user"), ""
        )
        h = hashlib.md5(first_user[:500].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(sample)
    return unique


def write_jsonl(samples: list[dict], path: Path) -> None:
    """Write samples as Apple JSONL format (just the messages array per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample["messages"], ensure_ascii=False) + "\n")


def main() -> None:
    random.seed(SEED)

    # --- Load datasets ---
    console.print("[bold]Loading datasets...[/]")
    ds_sft = load_from_disk(str(RAW_DIR / "typhoon_sft"))

    all_collected: list[dict] = []

    # --- Tier 1: autoif (Thai alignment) ---
    autoif_target = int(TARGET_SIZE * SOURCE_TARGETS["autoif"])
    autoif_rows = [r for r in ds_sft if (r.get("source") or "") == "autoif"]
    console.print(f"\n[bold]Tier 1: autoif ({len(autoif_rows):,} available)[/]")
    all_collected.extend(process_source(autoif_rows, "autoif", autoif_target))

    # --- Tier 1b: Sovereign (loaded from individual split files) ---
    sovereign_target = int(TARGET_SIZE * SOURCE_TARGETS["sovereign"])
    sovereign_rows: list[dict] = []

    # NitiBench SFT — has "messages" column (list of dicts with content/role)
    niti_path = RAW_DIR / "typhoon_sovereign_nitibench_sft"
    if niti_path.exists():
        ds_niti = load_from_disk(str(niti_path))
        sovereign_rows.extend(list(ds_niti))
        console.print(f"  Loaded {len(ds_niti):,} rows from nitibench_sft")
    else:
        console.print("  [yellow]Skipping nitibench_sft — not found[/]")

    # MIRAGE — has "prompt" column instead of "messages", normalize to "messages"
    mirage_path = RAW_DIR / "typhoon_sovereign_mirage"
    if mirage_path.exists():
        ds_mirage = load_from_disk(str(mirage_path))
        for row in ds_mirage:
            if "prompt" in row and row["prompt"]:
                sovereign_rows.append({"messages": row["prompt"]})
        console.print(f"  Loaded {len(ds_mirage):,} rows from mirage (prompt→messages)")
    console.print(f"\n[bold]Tier 1b: sovereign ({len(sovereign_rows):,} available)[/]")
    all_collected.extend(process_source(sovereign_rows, "sovereign", sovereign_target))

    # --- Tier 2: Tulu/general SFT (Thai-containing only) ---
    # Prioritize sources most likely to contain Thai: aya_100k, wildchat_100k, flan
    tulu_target = TARGET_SIZE - len(all_collected)
    tulu_rows = [r for r in ds_sft if (r.get("source") or "") != "autoif"]
    console.print(f"\n[bold]Tier 2: Tulu SFT Thai-filtered ({len(tulu_rows):,} available, need {tulu_target:,})[/]")
    all_collected.extend(
        process_source(tulu_rows, "tulu_sft", tulu_target, require_thai_filter=True)
    )

    # --- Deduplicate ---
    console.print(f"\n[bold]Before dedup:[/] {len(all_collected):,}")
    all_collected = deduplicate(all_collected)
    console.print(f"[bold]After dedup:[/] {len(all_collected):,}")

    # --- Split 80/20 ---
    random.shuffle(all_collected)
    n_eval = int(len(all_collected) * 0.20)
    eval_set = all_collected[:n_eval]
    train_set = all_collected[n_eval:]

    # --- Write output ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.jsonl"
    eval_path = OUTPUT_DIR / "eval.jsonl"
    write_jsonl(train_set, train_path)
    write_jsonl(eval_set, eval_path)

    # --- Summary ---
    table = Table(title="Output Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Path")
    table.add_row("Train", f"{len(train_set):,}", str(train_path))
    table.add_row("Eval", f"{len(eval_set):,}", str(eval_path))
    console.print(table)

    # Source distribution in final dataset
    source_dist = Counter(s["source"] for s in all_collected)
    table2 = Table(title="Source Distribution (final)")
    table2.add_column("Source", style="cyan")
    table2.add_column("Count", justify="right")
    table2.add_column("%", justify="right")
    for src, count in source_dist.most_common():
        table2.add_row(src, f"{count:,}", f"{100 * count / len(all_collected):.1f}%")
    console.print(table2)

    console.print("\n[bold green]Done! Ready for training.[/]")


if __name__ == "__main__":
    main()
