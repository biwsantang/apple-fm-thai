"""Validate Apple JSONL dataset files before training.

Checks:
  - Valid JSON on each line
  - Each line is a list of message dicts with role + content
  - Roles are "system" (optional, position 0 only), "user", or "assistant"
  - Alternating user/assistant pattern (after optional system)
  - Starts with user (or system then user), ends with assistant
  - No empty content
  - Reports Thai vs English distribution
"""

import json
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "iteration_1"


def validate_file(path: Path) -> tuple[int, int, list[str], int, int]:
    """Validate a JSONL file. Returns (valid_count, error_count, error_messages, thai_count, total_turns)."""
    valid = 0
    errors: list[str] = []
    thai_count = 0
    english_count = 0
    system_count = 0
    total_turns = 0

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                messages = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON — {e}")
                continue

            # Must be a list
            if not isinstance(messages, list):
                errors.append(f"Line {i}: Expected list, got {type(messages).__name__}")
                continue

            # At least 2 messages (user + assistant)
            if len(messages) < 2:
                errors.append(f"Line {i}: Too few messages ({len(messages)})")
                continue

            # Validate each message
            line_ok = True
            for j, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    errors.append(f"Line {i}, msg {j}: Not a dict")
                    line_ok = False
                    break

                if "role" not in msg or "content" not in msg:
                    errors.append(f"Line {i}, msg {j}: Missing 'role' or 'content'")
                    line_ok = False
                    break

                if msg["role"] not in ("system", "user", "assistant"):
                    errors.append(f"Line {i}, msg {j}: Invalid role '{msg['role']}'")
                    line_ok = False
                    break

                # System role only allowed at position 0
                if msg["role"] == "system" and j != 0:
                    errors.append(f"Line {i}, msg {j}: System role only allowed at position 0")
                    line_ok = False
                    break

                if not isinstance(msg["content"], str) or len(msg["content"].strip()) < 1:
                    errors.append(f"Line {i}, msg {j}: Empty content")
                    line_ok = False
                    break

            if not line_ok:
                continue

            # Determine first turn index (skip system if present)
            first_turn = 1 if messages[0]["role"] == "system" else 0

            if messages[0]["role"] == "system":
                # Need at least system + user + assistant = 3
                if len(messages) < 3:
                    errors.append(f"Line {i}: System + too few turns ({len(messages)})")
                    continue
                system_count += 1

            # Must start with user (after optional system), end with assistant
            if messages[first_turn]["role"] != "user":
                errors.append(f"Line {i}: First turn (idx {first_turn}) is not 'user'")
                continue
            if messages[-1]["role"] != "assistant":
                errors.append(f"Line {i}: Doesn't end with 'assistant'")
                continue

            # Alternating user/assistant pattern (after optional system)
            alternating = True
            for j in range(first_turn, len(messages) - 1):
                if messages[j]["role"] == messages[j + 1]["role"]:
                    errors.append(f"Line {i}: Non-alternating at position {j}")
                    alternating = False
                    break
            if not alternating:
                continue

            # Check Thai content
            full_text = " ".join(m["content"] for m in messages)
            if THAI_RE.search(full_text):
                thai_count += 1
            else:
                english_count += 1

            total_turns += len(messages)
            valid += 1

    return valid, len(errors), errors, thai_count, total_turns, english_count, system_count


def main() -> None:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR

    files = sorted(directory.glob("*.jsonl"))
    if not files:
        console.print(f"[red]No .jsonl files found in {directory}[/]")
        sys.exit(1)

    all_ok = True

    for path in files:
        console.print(f"\n[bold]Validating {path.name}...[/]")
        valid, error_count, errors, thai_count, total_turns, english_count, system_count = validate_file(path)

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Valid rows", f"{valid:,}")
        table.add_row("Errors", f"{error_count:,}")
        table.add_row("Thai-containing", f"{thai_count:,} ({100 * thai_count / valid:.1f}%)" if valid else "N/A")
        table.add_row("English-only", f"{english_count:,} ({100 * english_count / valid:.1f}%)" if valid else "N/A")
        table.add_row("With system msg", f"{system_count:,} ({100 * system_count / valid:.1f}%)" if valid else "N/A")
        table.add_row("Avg turns/conversation", f"{total_turns / valid:.1f}" if valid else "N/A")
        console.print(table)

        if errors:
            all_ok = False
            console.print(f"[red]First 10 errors:[/]")
            for err in errors[:10]:
                console.print(f"  {err}")

    if all_ok:
        console.print("\n[bold green]All files valid![/]")
    else:
        console.print("\n[bold red]Validation found errors. Fix before training.[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
