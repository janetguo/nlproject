"""Evaluate model using pass@k metrics with error taxonomy on HumanEval and MBPP."""

import json
import time
import numpy as np
from typing import List, Dict
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from utils import ErrorType, calculate_test_pass_rate, get_error_summary, validate_code_syntax, load_humaneval


# Fixed sampling parameters for reproducibility
TEMPERATURE = 0.8
TOP_P = 0.95
RANDOM_SEED = 42


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.
    
    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k
    
    Returns:
        Probability that at least one of k samples passes
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def load_mbpp(file_path: str = "mbpp.jsonl") -> List[Dict]:
    """Load MBPP benchmark with test cases."""
    try:
        problems = []
        with open(file_path, 'r') as f:
            for line in f:
                problem = json.loads(line)
                problems.append(problem)
        return problems
    except FileNotFoundError:
        print(f"❌ MBPP not found at {file_path}")
        print("Download from: https://github.com/google-research/google-research/tree/master/mbpp")
        return []


def generate_code_samples(
    model,
    tokenizer,
    prompt: str,
    num_samples: int = 10,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_length: int = 1024,
    seed: int = RANDOM_SEED
) -> List[str]:
    """Generate multiple code samples for a given prompt with fixed randomness."""
    samples = []
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    for i in range(num_samples):
        # Use different seed per sample but deterministic
        torch.manual_seed(seed + i)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        samples.append(generated)
    
    return samples


def evaluate_model(
    model_path: str,
    benchmark: str = "humaneval",
    benchmark_path: str = None,
    num_samples: int = 10,
    k_values: List[int] = [1, 10],
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    seed: int = RANDOM_SEED
) -> Dict:
    """
    Evaluate model on benchmark using pass@k with error taxonomy.
    
    Args:
        model_path: Path to model (HuggingFace format or local)
        benchmark: "humaneval" or "mbpp"
        benchmark_path: Path to benchmark file
        num_samples: Number of samples to generate per problem
        k_values: List of k values for pass@k (e.g., [1, 10])
        temperature: Sampling temperature (fixed)
        top_p: Nucleus sampling parameter (fixed)
        seed: Random seed for reproducibility
    
    Returns:
        Dict with comprehensive evaluation results
    """
    print("="*60)
    print(f"Evaluating {model_path} on {benchmark.upper()}")
    print(f"Sampling params: temp={temperature}, top_p={top_p}, seed={seed}")
    print("="*60)
    
    # Load model
    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=".hf-cache"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=".hf-cache")
    
    # Load benchmark
    print(f"Loading {benchmark} benchmark...")
    if benchmark.lower() == "humaneval":
        problems = load_humaneval(benchmark_path or "humaneval.jsonl")
    elif benchmark.lower() == "mbpp":
        problems = load_mbpp(benchmark_path or "mbpp.jsonl")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    if not problems:
        print("❌ Failed to load benchmark!")
        return {}
    
    print(f"Loaded {len(problems)} problems")
    
    # Evaluate each problem
    results = []
    total_errors = {error_type: 0 for error_type in ErrorType}
    total_time = 0
    compile_count = 0

    problems = problems[:10]
    
    for problem in tqdm(problems, desc="Evaluating"):
        prompt = problem.get('prompt', '')
        test_cases = problem.get('test_cases', [])
        
        if not prompt or not test_cases:
            continue
        
        # Generate samples
        start_time = time.time()
        samples = generate_code_samples(
            model,
            tokenizer,
            prompt,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
            seed=seed
        )
        gen_time = time.time() - start_time
        
        # Test each sample
        correct_count = 0
        problem_errors = {error_type: 0 for error_type in ErrorType}
        
        for sample in samples:
            # Check syntax first
            if validate_code_syntax(sample):
                compile_count += 1
            
            # Run tests
            pass_rate, error_counts = calculate_test_pass_rate(sample, test_cases)
            
            # Aggregate errors
            for error_type, count in error_counts.items():
                problem_errors[error_type] += count
                total_errors[error_type] += count
            
            # Count if all tests passed
            if pass_rate == 1.0:
                correct_count += 1
        
        results.append({
            'problem_id': problem.get('task_id', problem.get('problem_id', 'unknown')),
            'total': num_samples,
            'correct': correct_count,
            'errors': problem_errors,
            'generation_time': gen_time
        })
        
        total_time += gen_time
    
    # Calculate pass@k for different k values
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    pass_at_k_results = {}
    
    for k in k_values:
        scores = []
        for result in results:
            n = result['total']
            c = result['correct']
            scores.append(pass_at_k(n, c, k))
        
        avg_score = np.mean(scores)
        pass_at_k_results[f'pass@{k}'] = avg_score
        print(f"pass@{k}: {avg_score:.3f}")
    
    # Error taxonomy
    total_samples = len(results) * num_samples
    print(f"\n{'='*60}")
    print("ERROR TAXONOMY")
    print("="*60)
    print(f"Total samples: {total_samples}")
    print(f"Compile rate: {compile_count / total_samples:.3f} ({compile_count}/{total_samples})")

    total_tests = sum(total_errors.values())
    
    for error_type in ErrorType:
        count = total_errors[error_type]
        if count > 0:
            pct = count / total_tests * 100
            print(f"{error_type.value:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Operational metrics
    print(f"\n{'='*60}")
    print("OPERATIONAL METRICS")
    print("="*60)
    print(f"Avg time per problem: {total_time / len(results):.2f}s")
    print(f"Total evaluation time: {total_time:.2f}s")
    print("="*60)
    
    # Compile full results dict
    evaluation_results = {
        'model': model_path,
        'benchmark': benchmark,
        'num_problems': len(results),
        'num_samples': num_samples,
        'sampling_params': {
            'temperature': temperature,
            'top_p': top_p,
            'seed': seed
        },
        'pass_at_k': pass_at_k_results,
        'error_taxonomy': {error_type.value: total_errors[error_type] for error_type in ErrorType},
        'compile_rate': compile_count / total_samples,
        'operational_metrics': {
            'avg_time_per_problem': total_time / len(results),
            'total_time': total_time
        },
        'detailed_results': results
    }
    
    return evaluation_results


def compare_models(
    baseline_path: str,
    finetuned_path: str,
    benchmark: str = "humaneval",
    benchmark_path: str = None,
    num_samples: int = 10,
    output_file: str = None
):
    """
    Compare baseline vs fine-tuned model with detailed analysis.
    
    Performs paired statistical test and reports improvements.
    """
    from scipy import stats
    
    print("\n" + "="*60)
    print("MODEL COMPARISON: Baseline vs Fine-tuned")
    print("="*60)
    
    print("\n[1/2] Evaluating BASELINE model...")
    baseline_results = evaluate_model(
        baseline_path,
        benchmark=benchmark,
        benchmark_path=benchmark_path,
        num_samples=num_samples
    )
    
    print("\n[2/2] Evaluating FINE-TUNED model...")
    finetuned_results = evaluate_model(
        finetuned_path,
        benchmark=benchmark,
        benchmark_path=benchmark_path,
        num_samples=num_samples
    )
    
    # Statistical comparison
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    
    for k_name in baseline_results['pass_at_k'].keys():
        baseline_scores = [pass_at_k(r['total'], r['correct'], int(k_name.split('@')[1])) 
                          for r in baseline_results['detailed_results']]
        finetuned_scores = [pass_at_k(r['total'], r['correct'], int(k_name.split('@')[1]))
                           for r in finetuned_results['detailed_results']]
        
        baseline_score = baseline_results['pass_at_k'][k_name]
        finetuned_score = finetuned_results['pass_at_k'][k_name]
        improvement = finetuned_score - baseline_score
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(finetuned_scores, baseline_scores)
        
        print(f"\n{k_name}:")
        print(f"  Baseline:   {baseline_score:.3f}")
        print(f"  Fine-tuned: {finetuned_score:.3f}")
        print(f"  Change:     {improvement:+.3f} ({improvement/baseline_score*100:+.1f}%)")
        print(f"  p-value:    {p_value:.4f} {'✓ significant' if p_value < 0.05 else '✗ not significant'}")
    
    # Error taxonomy comparison
    print(f"\n{'='*60}")
    print("ERROR REDUCTION")
    print("="*60)
    
    baseline_errors = baseline_results['error_taxonomy']
    finetuned_errors = finetuned_results['error_taxonomy']
    
    for error_type in ErrorType:
        error_key = error_type.value
        baseline_count = baseline_errors.get(error_key, 0)
        finetuned_count = finetuned_errors.get(error_key, 0)
        
        if baseline_count > 0 or finetuned_count > 0:
            reduction = baseline_count - finetuned_count
            reduction_pct = (reduction / baseline_count * 100) if baseline_count > 0 else 0
            print(f"{error_key:20s}: {baseline_count:4d} → {finetuned_count:4d} ({reduction_pct:+.1f}%)")
    
    # Compile rate comparison
    baseline_compile = baseline_results['compile_rate']
    finetuned_compile = finetuned_results['compile_rate']
    compile_improvement = finetuned_compile - baseline_compile
    
    print(f"\n{'='*60}")
    print("COMPILE RATE")
    print("="*60)
    print(f"Baseline:   {baseline_compile:.3f}")
    print(f"Fine-tuned: {finetuned_compile:.3f}")
    print(f"Change:     {compile_improvement:+.3f}")
    
    print("="*60)
    
    # Save comparison results
    if output_file:
        comparison = {
            'baseline': baseline_results,
            'finetuned': finetuned_results
        }
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate code generation models")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model to evaluate")
    parser.add_argument("--baseline", type=str,
                        help="Path to baseline model (for comparison)")
    parser.add_argument("--benchmark", type=str, default="humaneval",
                        choices=["humaneval", "mbpp"],
                        help="Benchmark to use")
    parser.add_argument("--benchmark-path", type=str,
                        help="Path to benchmark file")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples per problem")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Sampling temperature (default: {TEMPERATURE})")
    parser.add_argument("--top-p", type=float, default=TOP_P,
                        help=f"Nucleus sampling parameter (default: {TOP_P})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument("--k", type=int, nargs='+', default=[1, 10],
                        help="k values for pass@k (e.g., --k 1 10)")
    parser.add_argument("--output", type=str,
                        help="Output file for results")
    
    args = parser.parse_args()
    
    if args.baseline:
        # Comparison mode
        compare_models(
            baseline_path=args.baseline,
            finetuned_path=args.model,
            benchmark=args.benchmark,
            benchmark_path=args.benchmark_path,
            num_samples=args.samples,
            output_file=args.output
        )
    else:
        # Single model evaluation
        results = evaluate_model(
            model_path=args.model,
            benchmark=args.benchmark,
            benchmark_path=args.benchmark_path,
            num_samples=args.samples,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            k_values=args.k
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
