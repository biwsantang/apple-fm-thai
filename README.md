# apple-fm-thai

Fine-tune Apple's on-device Foundation Model (~3B params) for Thai language using the [Typhoon 2.5](https://huggingface.co/typhoon-ai/typhoon2.5-qwen3-4b) training dataset.

## Why

Apple Intelligence's Foundation Model has limited Thai capability. This project creates a LoRA adapter that teaches the model to understand and respond in Thai, using the same dataset that trained Typhoon 2.5 — Thailand's leading open Thai LLM.

## Dataset

**Sources:**
- [`typhoon-ai/typhoon-s-instruct-post-training`](https://huggingface.co/datasets/typhoon-ai/typhoon-s-instruct-post-training) — SFT split (357K rows)
- [`typhoon-ai/typhoon-s-sovereign-capability-dataset`](https://huggingface.co/datasets/typhoon-ai/typhoon-s-sovereign-capability-dataset) — Thai legal/domain (NitiBench + MIRAGE)

**Final composition (3,972 conversations):**

| Source | Count | % | Language |
|--------|-------|---|----------|
| autoif | 1,582 | 40% | Thai |
| sovereign | 800 | 20% | Thai |
| tulu_sft_en | 1,596 | 40% | English |

The 60/40 Thai-English mix follows research from Typhoon 2 showing that English data prevents catastrophic forgetting during language adaptation.

## Pipeline

| Script | Purpose |
|--------|---------|
| `scripts/01_download_dataset.py` | Download from HuggingFace |
| `scripts/02_explore_dataset.py` | Source distribution, Thai content analysis |
| `scripts/03_filter_and_convert.py` | Filter, convert to Apple JSONL, 80/20 split |
| `scripts/04_validate_dataset.py` | Validate format (roles, alternation, Thai/English) |
| `scripts/05_validate_tokens.py` | Validate with Apple's real SentencePiece tokenizer |
| `scripts/06_train.sh` | Launch adapter training |
| `scripts/07_export.sh` | Export to `.fmadapter` bundle (macOS only) |
| `scripts/08_generate.py` | Run inference on eval prompts |

## Quick Start

```bash
# Setup
uv sync

# Download and process data
uv run python scripts/01_download_dataset.py
uv run python scripts/03_filter_and_convert.py
uv run python scripts/05_validate_tokens.py --filter

# Train (requires GPU with 24GB+ VRAM or Mac with 32GB+ RAM)
# Download Apple toolkit first: https://developer.apple.com/apple-intelligence/foundation-models-adapter/
./scripts/06_train.sh

# Export (macOS only)
./scripts/07_export.sh

# Evaluate
uv run python scripts/08_generate.py
uv run python scripts/08_generate.py --no-adapter  # baseline comparison
```

## Training Config

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 3 | Small dataset, avoid overfitting |
| Learning rate | 1e-3 | Apple toolkit default |
| Effective batch | 16 | batch=4 x accumulation=4 |
| Sequence packing | Enabled | Mean 481 tokens vs 4096 max context |
| Precision | bf16-mixed | Default, good speed/accuracy balance |
| LoRA rank | 32 | Fixed by Apple (alpha=16) |

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Apple Adapter Training Toolkit](https://developer.apple.com/apple-intelligence/foundation-models-adapter/) (requires Apple Developer Program)
- GPU with 24GB+ VRAM for training, or Mac with 32GB+ Apple Silicon
- macOS for export step (`coremltools` requirement)

## Notes

- Thai is not officially supported by Apple Intelligence — this is experimental
- Adapters are tied to a specific OS version; must retrain per iOS/macOS release
- Apple's guardrails may over-trigger for non-English languages (known issue)

## License

Dataset: [ODC-BY](https://opendatacommons.org/licenses/by/1-0/) (Typhoon)
Code: MIT
