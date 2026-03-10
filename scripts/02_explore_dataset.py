"""Explore Typhoon datasets: source distribution, sample rows, message stats."""

import re
from collections import Counter
from pathlib import Path

from datasets import load_from_disk
from rich.console import Console
from rich.table import Table

console = Console()
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

THAI_RE = re.compile(r"[\u0E00-\u0E7F]")


def contains_thai(text: str) -> bool:
    return bool(THAI_RE.search(text))


def conversation_text(messages) -> str:
    """Extract text from messages, handling both dict and string elements."""
    if isinstance(messages, str):
        return messages
    parts = []
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content", "")
            if isinstance(c, str):
                parts.append(c)
        elif isinstance(m, str):
            parts.append(m)
    return " ".join(parts)


def explore_sft() -> None:
    sft_path = RAW_DIR / "typhoon_sft"
    if not sft_path.exists():
        console.print("[red]SFT dataset not found. Run 01_download_dataset.py first.[/]")
        return

    console.print("[bold]Loading SFT dataset...[/]")
    ds = load_from_disk(str(sft_path))
    console.print(f"Total rows: {len(ds):,}")
    console.print(f"Columns: {ds.column_names}\n")

    # Source distribution
    source_counts = Counter(ds["source"])
    table = Table(title="Source Distribution")
    table.add_column("Source", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")
    for source, count in source_counts.most_common():
        table.add_row(source, f"{count:,}", f"{100 * count / len(ds):.1f}%")
    console.print(table)

    # Message count stats
    msg_counts = [len(row["messages"]) for row in ds]
    console.print(f"\n[bold]Message count stats:[/]")
    console.print(f"  Min: {min(msg_counts)}, Max: {max(msg_counts)}, Mean: {sum(msg_counts) / len(msg_counts):.1f}")

    # Thai content analysis (sample 10K for speed)
    sample_size = min(10_000, len(ds))
    thai_count = 0
    for i in range(sample_size):
        text = conversation_text(ds[i]["messages"])
        if contains_thai(text):
            thai_count += 1
    console.print(f"\n[bold]Thai content (sampled {sample_size:,} rows):[/]")
    console.print(f"  Thai-containing: {thai_count:,} ({100 * thai_count / sample_size:.1f}%)")

    # Thai by source (sample)
    thai_by_source: dict[str, int] = Counter()
    total_by_source: dict[str, int] = Counter()
    for i in range(sample_size):
        row = ds[i]
        src = row.get("source", "unknown")
        total_by_source[src] += 1
        if contains_thai(conversation_text(row["messages"])):
            thai_by_source[src] += 1

    table2 = Table(title=f"Thai Content by Source (sampled {sample_size:,})")
    table2.add_column("Source", style="cyan")
    table2.add_column("Thai Rows", justify="right")
    table2.add_column("Total Rows", justify="right")
    table2.add_column("Thai %", justify="right")
    for src in sorted(total_by_source, key=total_by_source.get, reverse=True):
        thai = thai_by_source.get(src, 0)
        total = total_by_source[src]
        table2.add_row(src, f"{thai:,}", f"{total:,}", f"{100 * thai / total:.1f}%" if total else "N/A")
    console.print(table2)

    # Sample rows (one per unique message format type)
    console.print("\n[bold]Sample rows:[/]")
    # Find rows with autoif source and one with empty source
    sample_indices = []
    for i in range(min(1000, len(ds))):
        src = ds[i].get("source") or ""
        if src == "autoif" and not any((ds[j].get("source") or "") == "autoif" for j in sample_indices):
            sample_indices.append(i)
        elif src == "" and not any((ds[j].get("source") or "") == "" for j in sample_indices):
            sample_indices.append(i)
        elif src.startswith("ai2") and not any((ds[j].get("source") or "").startswith("ai2") for j in sample_indices):
            sample_indices.append(i)
        if len(sample_indices) >= 3:
            break

    for i in sample_indices:
        row = ds[i]
        msgs = row["messages"]
        console.print(f"\n--- Row {i} (source: '{row.get('source', '')}') ---")
        console.print(f"  messages type: {type(msgs).__name__}, len: {len(msgs)}")
        if isinstance(msgs, list) and msgs:
            first = msgs[0]
            console.print(f"  first element type: {type(first).__name__}")
            if isinstance(first, dict):
                for msg in msgs[:4]:
                    role = msg.get("role", "?")
                    content = str(msg.get("content", ""))[:150]
                    console.print(f"  [{role}]: {content}...")
            else:
                console.print(f"  first element: {str(first)[:200]}...")
        elif isinstance(msgs, str):
            console.print(f"  content: {msgs[:200]}...")


def explore_sovereign() -> None:
    console.print("\n[bold]Sovereign datasets:[/]")
    sovereign_names = ["typhoon_sovereign_nitibench_sft", "typhoon_sovereign_mirage"]
    found_any = False

    for name in sovereign_names:
        path = RAW_DIR / name
        if not path.exists():
            console.print(f"  [yellow]{name} not found[/]")
            continue

        found_any = True
        ds = load_from_disk(str(path))
        console.print(f"  {name}: {len(ds):,} rows, columns: {ds.column_names}")

        # Sample first row
        row = ds[0]
        console.print(f"  Sample row:")
        # Handle both "messages" and "prompt" column formats
        messages = row.get("messages") or row.get("prompt", [])
        for msg in messages[:4]:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            content = content[:150] if isinstance(content, str) else str(content)[:150]
            console.print(f"    [{role}]: {content}...")

    if not found_any:
        console.print("[red]No sovereign splits found. Run 01_download_dataset.py first.[/]")


def main() -> None:
    explore_sft()
    explore_sovereign()


if __name__ == "__main__":
    main()
