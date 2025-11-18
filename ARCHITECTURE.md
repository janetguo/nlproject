# DPO Code Hallucination System Architecture

## High-Level Flow (Ground Truth Test-Based)

```
┌─────────────────────────────────────────────────────────────┐
│                     1. DATA PREPARATION                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: Coding problems with GROUND TRUTH TEST CASES         │
│         (HumanEval, MBPP, or custom with tests)              │
│         ↓                                                     │
│  Split: 50% train / 20% val / 30% test (seed=42)            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              2. PREFERENCE TUPLE GENERATION                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  For each problem:                                           │
│                                                               │
│  ┌─────────────────────────────────────────────────┐        │
│  │ Generate n=10 responses using Llama              │        │
│  │ (temperature=0.8, fixed for reproducibility)     │        │
│  └─────────────────────────────────────────────────┘        │
│                      ↓                                        │
│  ┌─────────────────────────────────────────────────┐        │
│  │ For each response:                               │        │
│  │   Execute on GROUND TRUTH test cases            │        │
│  │   Score = test pass rate (0.0 to 1.0)          │        │
│  │                                                   │        │
│  │   Track errors:                                  │        │
│  │   - Syntax errors (doesn't parse)                │        │
│  │   - Runtime errors (crashes)                     │        │
│  │   - Timeouts (>5s)                               │        │
│  │   - Wrong outputs (runs but incorrect)          │        │
│  └─────────────────────────────────────────────────┘        │
│                      ↓                                        │
│  ┌─────────────────────────────────────────────────┐        │
│  │ Create preference pairs:                         │        │
│  │   For each pair of responses (i, j):             │        │
│  │     if |score_i - score_j| >= 0.1:              │        │
│  │       chosen = higher test pass rate             │        │
│  │       rejected = lower test pass rate            │        │
│  │       save tuple (prompt, chosen, rejected)      │        │
│  └─────────────────────────────────────────────────┘        │
│                                                               │
│  Output: dpo_preferences.json                                │
│          (NO API calls needed!)                              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      3. DPO TRAINING                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────┐        │
│  │ Load base model: CodeLlama 7B                    │        │
│  │ Load reference model: CodeLlama 7B (frozen)      │        │
│  │ Apply LoRA adapters (optional, recommended)      │        │
│  │ Set seed=42 for reproducibility                  │        │
│  └─────────────────────────────────────────────────┘        │
│                      ↓                                        │
│  ┌─────────────────────────────────────────────────┐        │
│  │ DPO Loss:                                        │        │
│  │                                                   │        │
│  │ L = -log σ(β * (log(π_θ/π_ref)(chosen) -        │        │
│  │                  log(π_θ/π_ref)(rejected)))      │        │
│  │                                                   │        │
│  │ where:                                           │        │
│  │   π_θ = model being trained                      │        │
│  │   π_ref = reference model (frozen)               │        │
│  │   β = 0.1 (temperature parameter)                │        │
│  └─────────────────────────────────────────────────┘        │
│                      ↓                                        │
│  Training loop (1-3 epochs)                                  │
│  Save config, logs, checkpoints                              │
│                                                               │
│  Output: Fine-tuned model                                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      4. EVALUATION                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  For each test problem:                                      │
│                                                               │
│  ┌─────────────────────────────────────────────────┐        │
│  │ Generate k=10 solutions                          │        │
│  │ (temp=0.8, top_p=0.95, seed=42)                │        │
│  │ Execute against ground truth test cases          │        │
│  │ Categorize errors + count correct (c)            │        │
│  └─────────────────────────────────────────────────┘        │
│                      ↓                                        │
│  ┌─────────────────────────────────────────────────┐        │
│  │ Calculate pass@k:                                │        │
│  │                                                   │        │
│  │   pass@k = 1 - C(n-c, k) / C(n, k)              │        │
│  │                                                   │        │
│  │ where n=10 samples, c=correct, k∈{1,10}         │        │
│  └─────────────────────────────────────────────────┘        │
│                      ↓                                        │
│  Metrics:                                                     │
│    - pass@1, pass@10 (functional correctness)                │
│    - Error taxonomy (syntax, runtime, timeout, wrong)        │
│    - Compile rate                                            │
│    - Operational metrics (time per problem)                  │
│    - Statistical significance (paired t-test, p<0.05)        │
│                                                               │
│  Compare: Baseline vs Fine-tuned                             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Advantages Over Consensus-Based Scoring

| Aspect | Consensus (Old) | Ground Truth (New) |
|--------|-----------------|-------------------|
| **Measures** | Agreement between implementations | Actual correctness |
| **Cost** | ~5-15 Claude API calls per function | 0 API calls |
| **Speed** | Slow (generate alternatives) | Fast (just execute) |
| **Systematic errors** | Vulnerable | Robust |
| **Reliability** | Overconfident on shared mistakes | Objective |

## Error Taxonomy

```
ErrorType:
├─ SYNTAX_ERROR     → Code doesn't parse
├─ RUNTIME_ERROR    → Code crashes during execution
├─ TIMEOUT          → Execution > 5 seconds
├─ WRONG_OUTPUT     → Runs but produces incorrect output
└─ SUCCESS          → All tests passed ✓
```

This categorization helps distinguish:
- **True hallucinations** (uses fake APIs, nonsensical logic)
- **Simple bugs** (off-by-one, wrong condition)
- **Performance issues** (inefficient algorithm)

## Reproducibility Controls

All experiments use **fixed parameters**:
- Random seed: 42 (for all RNGs: Python, NumPy, PyTorch)
- Temperature: 0.8 (for generation)
- Top-p: 0.95 (nucleus sampling)
- Dataset splits: 50/20/30 (train/val/test)

## File Structure & Dependencies

```
utils.py
  ├─ ErrorType enum
  ├─ execute_code_with_test()
  ├─ calculate_test_pass_rate()
  └─ validate_code_syntax()
       ↓
generate_preferences.py
  ├─ PreferenceGenerator class
  ├─ Uses: Llama inference (TODO: implement)
  ├─ Uses: Ground truth tests from dataset
  └─ Outputs: dpo_preferences.json
       ↓
train_dpo.py
  ├─ Uses: HuggingFace Transformers + TRL
  ├─ set_seed() for reproducibility
  ├─ Input: dpo_preferences.json
  └─ Output: Fine-tuned model + config
       ↓
evaluate.py
  ├─ Uses: Fine-tuned model
  ├─ Uses: HumanEval/MBPP ground truth tests
  ├─ Categorizes errors with ErrorType
  ├─ Statistical testing (scipy.stats)
  └─ Output: Comprehensive results JSON
```

## Experimental Design

### Training
- Train on 50% of dataset (e.g., 80 problems from HumanEval)
- Generate 10 responses per problem
- Create ~C(10,2) = 45 preference pairs per problem (filtered by score diff)
- Total: ~3,600 preference pairs

### Validation
- Use 20% for hyperparameter tuning
- Tune: learning rate, beta, epochs
- Don't report these results in final paper

### Testing
- Hold out 30% for final evaluation
- **Never** use during training or tuning
- Report only these results

### Statistical Testing
```python
from scipy.stats import ttest_rel

baseline_scores = [pass@k for each problem]
finetuned_scores = [pass@k for each problem]

t_stat, p_value = ttest_rel(finetuned_scores, baseline_scores)

if p_value < 0.05:
    print("✓ Improvement is statistically significant")
```

## Resource Requirements

| Stage | GPU | Time | Cost |
|-------|-----|------|------|
| Preference Gen | N/A | 30min-4h | $0 |
| Training (Full) | A100 40GB | 3-4h | $10-20 |
| Training (LoRA) | A100 16GB | 3-4h | $3-6 |
| Evaluation | Any 16GB+ | 1-2h | Free |

## Timeline (4 Weeks)

**Week 1:**
- Implement Llama inference
- Generate preferences on 20 problems (test pipeline)
- Verify error taxonomy works

**Week 2:**
- Scale to 80 training problems
- Train baseline + DPO models
- Initial evaluation on validation set

**Week 3:**
- Tune hyperparameters on validation set
- Final training run with best params
- Full evaluation on test set

**Week 4:**
- Statistical analysis
- Error categorization analysis
- Write report + make poster

## Expected Results

Based on similar work, expect:
- pass@1 improvement: +10-20% relative
- pass@10 improvement: +5-15% relative
- Compile rate: +3-5% absolute
- Error reduction: -15-25% in runtime/wrong output

Improvements should be **statistically significant** (p < 0.05) on HumanEval test set.
