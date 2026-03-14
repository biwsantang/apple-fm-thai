"""Validate token counts using Apple's real SentencePiece tokenizer.

Loads the AFMTokenizer from the toolkit and tokenizes every conversation
to get actual token counts (including chat template overhead).
Reports distribution and flags conversations exceeding the threshold.
"""

import json
import sys
from pathlib import Path

import tamm
from tamm.tokenizers.afm import AFMTokenizer
from tamm.preprocessors.instruct_lm.afm import AFMChatTemplateV6Preprocessor
from rich.console import Console
from rich.table import Table

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "apple-fm-toolkit" / "assets"
DEFAULT_DIR = PROJECT_ROOT / "data" / "processed" / "iteration_1"

MAX_CONTEXT = 4096
WARN_THRESHOLD = 3500


def load_tokenizer() -> AFMTokenizer:
    config_path = ASSETS_DIR / "tokenizer-config.json"
    with config_path.open() as f:
        config = tamm.utils.json.load(f)
    vocab_path = ASSETS_DIR / "tokenizer.model"
    tokenizer = config.create_tokenizer(vocab_path=vocab_path)
    if not hasattr(tokenizer, "ignore_id"):
        tokenizer.ignore_id = -100
    return tokenizer


def count_tokens(messages: list[dict], tokenizer: AFMTokenizer) -> int:
    """Tokenize a conversation and return actual token count."""
    has_system = messages[0]["role"] == "system" if messages else False
    preprocessor = AFMChatTemplateV6Preprocessor(
        tokenizer_spec=tokenizer,
        generate_prefix=False,
        max_sequence_length=None,
        prepend_system_message=not has_system,
    )
    sample = preprocessor(messages)
    return len(sample.input_ids)


def validate_file(path: Path, tokenizer: AFMTokenizer) -> dict:
    """Validate token counts for all conversations in a JSONL file."""
    counts = []
    over_warn = []
    over_max = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)
            n_tokens = count_tokens(messages, tokenizer)
            counts.append(n_tokens)

            if n_tokens > MAX_CONTEXT:
                over_max.append((i, n_tokens))
            elif n_tokens > WARN_THRESHOLD:
                over_warn.append((i, n_tokens))

    counts.sort()
    n = len(counts)
    return {
        "total": n,
        "min": counts[0] if counts else 0,
        "max": counts[-1] if counts else 0,
        "mean": sum(counts) / n if n else 0,
        "median": counts[n // 2] if n else 0,
        "p90": counts[int(n * 0.90)] if n else 0,
        "p95": counts[int(n * 0.95)] if n else 0,
        "p99": counts[int(n * 0.99)] if n else 0,
        "over_warn": over_warn,
        "over_max": over_max,
        "counts": counts,
    }


def filter_file(path: Path, tokenizer: AFMTokenizer, max_tokens: int = MAX_CONTEXT) -> int:
    """Remove conversations exceeding max_tokens. Rewrites file in-place. Returns count removed."""
    kept = []
    removed = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)
            n_tokens = count_tokens(messages, tokenizer)
            if n_tokens <= max_tokens:
                kept.append(line)
            else:
                removed += 1

    with open(path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")

    return removed


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    directory = Path(args[0]) if args else DEFAULT_DIR
    do_filter = "--filter" in flags

    if not ASSETS_DIR.exists():
        console.print(f"[red]Apple toolkit assets not found at {ASSETS_DIR}[/]")
        console.print("Download the toolkit and place it at apple-fm-toolkit/")
        sys.exit(1)

    console.print("[bold]Loading Apple tokenizer...[/]")
    tokenizer = load_tokenizer()
    console.print(f"[green]Tokenizer loaded[/]")

    files = sorted(directory.glob("*.jsonl"))
    if not files:
        console.print(f"[red]No .jsonl files found in {directory}[/]")
        sys.exit(1)

    all_ok = True

    for path in files:
        # Filter first if requested
        if do_filter:
            removed = filter_file(path, tokenizer)
            if removed:
                console.print(f"[yellow]Filtered {removed} conversations exceeding {MAX_CONTEXT} tokens from {path.name}[/]")

        console.print(f"\n[bold]Tokenizing {path.name}...[/]")
        result = validate_file(path, tokenizer)

        table = Table(title=f"Token Distribution — {path.name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Total conversations", f"{result['total']:,}")
        table.add_row("Min tokens", f"{result['min']:,}")
        table.add_row("Max tokens", f"{result['max']:,}")
        table.add_row("Mean tokens", f"{result['mean']:.0f}")
        table.add_row("Median tokens", f"{result['median']:,}")
        table.add_row("P90", f"{result['p90']:,}")
        table.add_row("P95", f"{result['p95']:,}")
        table.add_row("P99", f"{result['p99']:,}")
        table.add_row(f"Over {WARN_THRESHOLD} (warn)", f"{len(result['over_warn']):,}")
        table.add_row(f"Over {MAX_CONTEXT} (EXCEEDS MAX)", f"{len(result['over_max']):,}")
        console.print(table)

        if result["over_max"]:
            all_ok = False
            console.print(f"[red]Lines exceeding {MAX_CONTEXT} tokens:[/]")
            for line_num, n_tok in result["over_max"][:20]:
                console.print(f"  Line {line_num}: {n_tok:,} tokens")

        if result["over_warn"]:
            console.print(f"[yellow]Lines between {WARN_THRESHOLD}-{MAX_CONTEXT} tokens:[/]")
            for line_num, n_tok in result["over_warn"][:10]:
                console.print(f"  Line {line_num}: {n_tok:,} tokens")

    if all_ok:
        console.print(f"\n[bold green]All conversations under {MAX_CONTEXT} tokens![/]")
    else:
        console.print(f"\n[bold red]Some conversations exceed {MAX_CONTEXT} tokens. Run with --filter to remove them.[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
