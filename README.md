# DPO for Code Hallucination Reduction

Implementation of Direct Preference Optimization (DPO) using **ground truth test pass rates** for reducing hallucinations in code generation LLMs.

## Approach

Uses **test-based truthfulness scoring** where:
1. Generate multiple responses from base model
2. Score each response by **test pass rate** on ground truth tests
3. Create preference pairs: higher pass rate = chosen, lower = rejected
4. Train using DPO

**Key difference from consensus methods:** We use actual correctness (test pass rate) instead of model confidence or agreement between implementations.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Implement Llama Inference

You need to add Llama inference to `generate_preferences.py` (line 24). See README for options: HuggingFace, vLLM, or Replicate API.

### 3. Generate Preferences

```bash
# Test with dummy data
python generate_preferences.py --dataset dummy --output dpo_preferences.json

# Use HumanEval
python generate_preferences.py --dataset humaneval --dataset-path humaneval.jsonl --output dpo_preferences.json
```

### 4. Train

```bash
# LoRA (recommended, needs 16GB VRAM)
python train_dpo.py --dataset dpo_preferences.json --output ./dpo_model_lora --lora --epochs 3

# Full (needs 40GB VRAM)
python train_dpo.py --dataset dpo_preferences.json --output ./dpo_model --epochs 1
```

### 5. Evaluate

```bash
# Single model
python evaluate.py --model ./dpo_model --benchmark humaneval --samples 10

# Compare baseline vs fine-tuned
python evaluate.py --model ./dpo_model --baseline codellama/CodeLlama-7b-hf --benchmark humaneval
```

## Key Features

✓ Uses ground truth tests (not consensus)
✓ Error taxonomy (syntax, runtime, timeout, wrong output)
✓ Statistical significance testing (paired t-test)
✓ Full reproducibility (fixed seeds, locked sampling params)
✓ Compile rate tracking
✓ Operational metrics

## Cost

- **Preference generation**: $0 (no API calls!)
- **Training**: ~$3-6 per epoch (LoRA on Modal/Colab)
- **Evaluation**: Free (local)

## Files

- `generate_preferences.py` - Generate preferences from ground truth
- `train_dpo.py` - Train with DPO
- `evaluate.py` - Evaluate with error taxonomy
- `utils.py` - Execution and error categorization
- `example.py` - Example usage

See full documentation in this README for dataset format, parameters, troubleshooting, and expected results.
