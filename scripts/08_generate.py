"""Run inference on evaluation prompts using trained adapter.

Loads the base model + adapter checkpoint, runs each prompt from
eval/prompts.jsonl, and saves responses to eval/results.jsonl.

Usage:
    uv run python scripts/08_generate.py
    uv run python scripts/08_generate.py --checkpoint adapter/iteration_1/adapter-epoch2.pt
    uv run python scripts/08_generate.py --no-adapter  # baseline without adapter
"""

import json
import re
import sys
import time
from pathlib import Path

import torch

# Add toolkit to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
TOOLKIT_DIR = PROJECT_DIR / "apple-fm-toolkit"
sys.path.insert(0, str(TOOLKIT_DIR))

from examples.utils import (
    load_tokenizer,
    load_base_model,
    get_device,
    PrecisionConfig,
    autocast,
)
from examples.generate import GenerationConfiguration, _parse_outputs
from examples.data import create_dataloader, BatchKey
from rich.console import Console
from rich.table import Table

console = Console()

THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
DEFAULT_CHECKPOINT = PROJECT_DIR / "adapter" / "iteration_1" / "adapter-final.pt"
PROMPTS_PATH = PROJECT_DIR / "eval" / "prompts.jsonl"
RESULTS_PATH = PROJECT_DIR / "eval" / "results.jsonl"


def load_prompts(path: Path) -> list[list[dict]]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


@torch.inference_mode()
def generate_response(
    messages: list[dict],
    model,
    tokenizer,
    device: torch.device,
    gen_config: GenerationConfiguration,
) -> tuple[str, float]:
    """Generate a single response. Returns (response_text, elapsed_seconds)."""
    loader = create_dataloader(
        data=[messages],
        tokenizer=tokenizer,
        batch_size=1,
        padding_side="left",
    )
    batch = next(iter(loader))
    input_ids = batch[BatchKey.INPUT].to(device)

    start = time.perf_counter()
    with autocast(device=device, dtype=gen_config.autocast_dtype):
        generator = gen_config.create_generator(model, None, tokenizer)
        output = generator.generate(
            input_ids=input_ids,
            max_new_tokens=gen_config.max_new_tokens,
        )
    elapsed = time.perf_counter() - start

    response = _parse_outputs(tokenizer, output.token_ids)[0]
    return response, elapsed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on eval prompts")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to adapter checkpoint",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Run without adapter (baseline)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per response",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=str(PROMPTS_PATH),
        help="Path to prompts JSONL file",
    )
    args = parser.parse_args()

    # Setup
    device = get_device()
    precision = PrecisionConfig("bf16-mixed")
    console.print(f"[bold]Device:[/] {device}")

    # Load tokenizer
    console.print("[bold]Loading tokenizer...[/]")
    tokenizer = load_tokenizer()

    # Load model
    checkpoint = None if args.no_adapter else args.checkpoint
    if checkpoint and not Path(checkpoint).exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/]")
        console.print("Run training first or use --no-adapter for baseline.")
        sys.exit(1)

    label = "baseline (no adapter)" if args.no_adapter else f"adapter: {checkpoint}"
    console.print(f"[bold]Loading model ({label})...[/]")
    model = load_base_model(
        dtype=precision.model_dtype,
        device=device,
        checkpoint_path=checkpoint,
    ).eval()

    # Generation config
    gen_config = GenerationConfiguration(
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    # Load prompts
    prompts = load_prompts(Path(args.prompts))
    console.print(f"\n[bold]Running {len(prompts)} prompts...[/]\n")

    # Generate responses
    results = []
    thai_responses = 0
    total_time = 0.0

    for i, messages in enumerate(prompts):
        prompt_text = messages[-1]["content"]  # Last user message
        is_thai_prompt = bool(THAI_RE.search(prompt_text))

        response, elapsed = generate_response(
            messages, model, tokenizer, device, gen_config
        )
        total_time += elapsed

        is_thai_response = bool(THAI_RE.search(response))
        if is_thai_response:
            thai_responses += 1

        result = {
            "prompt": prompt_text,
            "response": response,
            "is_thai_prompt": is_thai_prompt,
            "is_thai_response": is_thai_response,
            "tokens_generated": len(response),
            "time_seconds": round(elapsed, 2),
        }
        results.append(result)

        # Print progress
        lang_tag = "[cyan]TH[/]" if is_thai_prompt else "[green]EN[/]"
        resp_lang = "[cyan]TH[/]" if is_thai_response else "[green]EN[/]"
        console.print(
            f"  [{i+1}/{len(prompts)}] {lang_tag}→{resp_lang} ({elapsed:.1f}s) "
            f"{prompt_text[:60]}..."
        )
        console.print(f"    {response[:120]}...\n")

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary
    thai_prompts = sum(1 for r in results if r["is_thai_prompt"])
    en_prompts = len(results) - thai_prompts
    thai_prompt_thai_resp = sum(
        1 for r in results if r["is_thai_prompt"] and r["is_thai_response"]
    )
    en_prompt_en_resp = sum(
        1 for r in results if not r["is_thai_prompt"] and not r["is_thai_response"]
    )

    table = Table(title="Inference Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total prompts", str(len(results)))
    table.add_row("Thai prompts", str(thai_prompts))
    table.add_row("English prompts", str(en_prompts))
    table.add_row(
        "Thai→Thai (correct language)",
        f"{thai_prompt_thai_resp}/{thai_prompts} ({100*thai_prompt_thai_resp/thai_prompts:.0f}%)"
        if thai_prompts
        else "N/A",
    )
    table.add_row(
        "English→English (retained)",
        f"{en_prompt_en_resp}/{en_prompts} ({100*en_prompt_en_resp/en_prompts:.0f}%)"
        if en_prompts
        else "N/A",
    )
    table.add_row("Avg time/response", f"{total_time/len(results):.1f}s")
    table.add_row("Results saved to", str(RESULTS_PATH))
    console.print(table)


if __name__ == "__main__":
    main()
