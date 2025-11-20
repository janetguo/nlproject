"""Generate preference pairs for DPO training using ground truth tests."""

import json
from typing import List, Dict, Tuple
from tqdm import tqdm
from utils import calculate_test_pass_rate, ErrorType
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

class PreferenceGenerator:
    """Generates DPO preference tuples using ground truth test pass rates."""
    
    def __init__(self, base_model: str = "codellama/CodeLlama-7b-Instruct-hf"):
        self.base_model = base_model
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, cache_dir=".hf-cache/", torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, cache_dir=".hf-cache/")
    
    def generate_response_with_llama(self, prompt: str, temperature: float = 0.8) -> str:
        """
        Generate code response using base Llama model.
        
        TODO: Replace with actual Llama API call or local inference.
        
        Options:
        1. HuggingFace local:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(self.base_model)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=1024, temperature=temperature)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        2. vLLM (faster):
            from vllm import LLM, SamplingParams
            llm = LLM(model=self.base_model)
            params = SamplingParams(temperature=temperature, max_tokens=1024)
            outputs = llm.generate([prompt], params)
            return outputs[0].outputs[0].text
        
        3. Replicate API:
            import replicate
            output = replicate.run("meta/codellama-7b:...", 
                                   input={"prompt": prompt})
            return output
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_length=1024, temperature=temperature, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    
    def generate_preference_data_for_problem(
        self,
        problem: Dict,
        num_responses: int = 10,
        temperature: float = 0.8,
        min_score_diff: float = 0.1
    ) -> List[Dict]:
        """
        Generate DPO preference tuples for a single problem using ground truth tests.
        
        Args:
            problem: Dict with:
                - 'prompt': str (the coding problem)
                - 'test_cases': List[Dict] with 'input' and 'output' keys
                - 'problem_id': str (optional, for logging)
            num_responses: Number of responses to generate from base model
            temperature: Sampling temperature
            min_score_diff: Minimum score difference to create preference pair
        
        Returns:
            List of preference tuples: [{
                'prompt': str,
                'chosen': str,  # Higher test pass rate
                'rejected': str,  # Lower test pass rate
                'chosen_score': float,
                'rejected_score': float,
                'score_diff': float
            }]
        """
        prompt = problem['prompt']
        test_cases = problem.get('test_cases', [])
        problem_id = problem.get('problem_id', 'unknown')
        
        if not test_cases:
            print(f"  ⚠ No test cases for problem {problem_id}, skipping")
            return []
        
        print(f"\n{'='*60}")
        print(f"Processing problem: {problem_id}")
        print(f"{'='*60}")
        
        # Step 1: Generate n responses from base model
        print(f"\nGenerating {num_responses} responses...")
        responses = []
        for i in range(num_responses):
            try:
                response = self.generate_response_with_llama(prompt, temperature)
                responses.append(response)
                print(f"  ✓ Generated response {i+1}/{num_responses}")
            except NotImplementedError:
                print("  ✗ Llama inference not implemented - using placeholder")
                # Placeholder for testing structure
                responses.append(f"# Placeholder response {i}\ndef solution(x):\n    return x")
        
        # Step 2: Score each response on ground truth tests
        print("\nScoring responses on ground truth tests...")
        scored_responses = []
        
        for idx, response_code in enumerate(responses):
            print(f"\n  Response {idx+1}/{len(responses)}:")
            print(response_code)
            
            # Calculate test pass rate
            pass_rate, error_counts = calculate_test_pass_rate(response_code, test_cases)
            
            # Log error breakdown
            print(f"    Pass rate: {pass_rate:.3f}")
            print(f"    Errors: ", end="")
            for error_type, count in error_counts.items():
                if count > 0 and error_type != ErrorType.SUCCESS:
                    print(f"{error_type.value}={count} ", end="")
            print()
            
            scored_responses.append({
                'code': response_code,
                'score': pass_rate,
                'error_counts': error_counts
            })
        
        # Step 3: Create preference pairs
        print("\nCreating preference pairs...")
        preference_pairs = []
        
        for i in range(len(scored_responses)):
            for j in range(i+1, len(scored_responses)):
                resp_i = scored_responses[i]
                resp_j = scored_responses[j]
                
                score_diff = abs(resp_i['score'] - resp_j['score'])
                
                # Skip if scores are too similar (weak signal for DPO)
                if score_diff < min_score_diff:
                    continue
                
                # Higher score is chosen
                if resp_i['score'] > resp_j['score']:
                    chosen, rejected = resp_i, resp_j
                else:
                    chosen, rejected = resp_j, resp_i
                
                preference_pairs.append({
                    'prompt': prompt,
                    'chosen': chosen['code'],
                    'rejected': rejected['code'],
                    'chosen_score': chosen['score'],
                    'rejected_score': rejected['score'],
                    'score_diff': score_diff
                })
        
        print(f"Created {len(preference_pairs)} preference pairs")
        print(f"  (with score diff >= {min_score_diff})")
        
        return preference_pairs
    
    def generate_dataset(
        self,
        problems: List[Dict],
        output_file: str,
        **kwargs
    ):
        """
        Generate complete DPO dataset from list of problems.
        
        Args:
            problems: List of dicts with 'prompt' and 'test_cases' keys
            output_file: Path to save JSON dataset
            **kwargs: Additional args for generate_preference_data_for_problem
        """
        all_preferences = []
        
        print(f"\n{'='*60}")
        print(f"Starting preference generation for {len(problems)} problems")
        print(f"{'='*60}")
        
        for problem in tqdm(problems, desc="Processing problems"):
            try:
                preferences = self.generate_preference_data_for_problem(problem, **kwargs)
                all_preferences.extend(preferences)
            except Exception as e:
                print(f"\n❌ Error processing problem: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save dataset
        print(f"\n{'='*60}")
        print(f"Saving dataset...")
        with open(output_file, 'w') as f:
            json.dump(all_preferences, f, indent=2)
        
        # Print statistics
        if all_preferences:
            avg_score_diff = sum(p['score_diff'] for p in all_preferences) / len(all_preferences)
            avg_chosen_score = sum(p['chosen_score'] for p in all_preferences) / len(all_preferences)
            avg_rejected_score = sum(p['rejected_score'] for p in all_preferences) / len(all_preferences)
            
            print(f"\n{'='*60}")
            print(f"DATASET STATISTICS")
            print(f"{'='*60}")
            print(f"Total preference pairs: {len(all_preferences)}")
            print(f"Avg chosen score: {avg_chosen_score:.3f}")
            print(f"Avg rejected score: {avg_rejected_score:.3f}")
            print(f"Avg score difference: {avg_score_diff:.3f}")
            print(f"Saved to: {output_file}")
            print(f"{'='*60}")
        else:
            print(f"\n⚠ No preference pairs generated!")


def load_humaneval(file_path: str = "humaneval.jsonl") -> List[Dict]:
    """
    Load HumanEval dataset from HuggingFace.

    Dataset: openai/human-eval
    Each record has:
        - task_id
        - prompt
        - entry_point
        - test        (full test code)
        - canonical_solution

    Returns a list of problems in the same format as the original loader.
    """
    ds = load_dataset("openai_humaneval")["test"]

    problems = []

    for item in ds:
        problems.append({
            "problem_id": item["task_id"],
            "prompt": item["prompt"],
            "test_code": item.get("test", ""),
            "entry_point": item.get("entry_point", ""),
            "canonical_solution": item.get("canonical_solution", ""),
            "test_cases": item.get("test", "")
        })

    return problems


def load_mbpp(file_path: str = "mbpp.jsonl") -> List[Dict]:
    """
    Load MBPP dataset.
    
    Expected format:
    {
        "task_id": 1,
        "text": "Write a function...",
        "code": "def solution(...)...",
        "test_list": ["assert solution(1) == 2", ...]
    }
    """
    problems = []
    
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            
            problems.append({
                'problem_id': f"MBPP/{item['task_id']}",
                'prompt': item['text'],
                'test_list': item.get('test_list', []),
                'code': item.get('code', ''),
                'test_cases': []  # Will be populated
            })
    
    return problems


def split_dataset(
    problems: List[Dict],
    train_ratio: float = 0.05,
    val_ratio: float = 0.2,
    test_ratio: float = 0.3,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train/val/test with fixed seed for reproducibility.
    
    Args:
        problems: List of problems
        train_ratio: Fraction for training (DPO preference generation)
        val_ratio: Fraction for validation (hyperparameter tuning)
        test_ratio: Fraction for final evaluation (held-out)
        seed: Random seed for reproducibility
    
    Returns:
        (train_problems, val_problems, test_problems)
    """
    import random
    random.seed(seed)
    
    # Shuffle
    shuffled = problems.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    print(f"\nDataset split (seed={seed}):")
    print(f"  Train: {len(train)} problems ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val)} problems ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test)} problems ({test_ratio*100:.0f}%)")
    
    return train, val, test


if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate DPO preferences from ground truth tests")
    parser.add_argument("--dataset", type=str, choices=["humaneval", "mbpp", "dummy"],
                        default="dummy", help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset file")
    parser.add_argument("--output", type=str, default="dpo_preferences.json",
                        help="Output file for preferences")
    parser.add_argument("--n-responses", type=int, default=10,
                        help="Number of responses per problem")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--min-score-diff", type=float, default=0.1,
                        help="Minimum score difference for preference pairs")
    parser.add_argument("--train-only", action="store_true",
                        help="Only use training split")
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == "dummy":
        print("Using dummy problems for testing...")
        problems = [
            {
                'problem_id': 'test_1',
                'prompt': 'def factorial(n):\n    """Compute factorial of n"""\n    ',
                'test_cases': [
                    {'input': 0, 'output': 1},
                    {'input': 1, 'output': 1},
                    {'input': 5, 'output': 120},
                    {'input': 10, 'output': 3628800}
                ]
            },
            {
                'problem_id': 'test_2',
                'prompt': 'def is_palindrome(s):\n    """Check if string is palindrome"""\n    ',
                'test_cases': [
                    {'input': 'racecar', 'output': True},
                    {'input': 'hello', 'output': False},
                    {'input': 'A man a plan a canal Panama', 'output': False},
                    {'input': '', 'output': True}
                ]
            }
        ]
    elif args.dataset == "humaneval":
        if not args.dataset_path:
            args.dataset_path = "humaneval.jsonl"
        print(f"Loading HumanEval from {args.dataset_path}...")
        problems = load_humaneval(args.dataset_path)
    elif args.dataset == "mbpp":
        if not args.dataset_path:
            args.dataset_path = "mbpp.jsonl"
        print(f"Loading MBPP from {args.dataset_path}...")
        problems = load_mbpp(args.dataset_path)
    
    # Split dataset
    if args.train_only or args.dataset == "dummy":
        train_problems = problems
    else:
        train_problems, val_problems, test_problems = split_dataset(problems)
    
    # Initialize generator
    generator = PreferenceGenerator()
    
    # Generate preferences
    generator.generate_dataset(
        problems=train_problems,
        output_file=args.output,
        num_responses=args.n_responses,
        temperature=args.temperature,
        min_score_diff=args.min_score_diff
    )
