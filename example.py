"""Example walkthrough of the DPO pipeline."""

import os
from generate_preferences import PreferenceGenerator, load_problems_from_dataset

# Example: How to use the system

def example_minimal_run():
    """Minimal example with 2 problems."""
    
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not API_KEY:
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        return
    
    # Create test problems
    problems = [
        {
            'problem_id': 'factorial',
            'prompt': '''Write a function compute_factorial(n) that computes n!
Requirements:
- Use helper functions where appropriate
- Include docstrings for all functions'''
        },
        {
            'problem_id': 'palindrome',
            'prompt': '''Write a function is_palindrome(s) that checks if string is palindrome.
Requirements:
- Use helper functions where appropriate
- Include docstrings for all functions'''
        }
    ]
    
    # Initialize generator
    print("Initializing preference generator...")
    generator = PreferenceGenerator(api_key=API_KEY)
    
    # Generate preferences (SMALL NUMBERS for testing)
    print("\nGenerating DPO preferences...")
    print("Note: This will make many API calls. Cost estimate: ~$5-10")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    generator.generate_dataset(
        problems=problems,
        output_file='test_preferences.json',
        num_responses=3,  # Generate 3 responses per problem
        num_alternative_impls=3,  # Generate 3 alternatives per function
        num_test_cases=5  # 5 test cases per function
    )
    
    print("\n✓ Preferences saved to test_preferences.json")
    print("\nNext steps:")
    print("1. Implement Llama inference in generate_preferences.py")
    print("2. Run: python train_dpo.py --dataset test_preferences.json --lora")
    print("3. Run: python evaluate.py --model ./dpo_model_lora --baseline codellama/CodeLlama-7b-hf")


def example_with_humaneval():
    """Example using HumanEval dataset."""
    
    print("To use HumanEval:")
    print("\n1. Download dataset:")
    print("   wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz")
    print("   gunzip HumanEval.jsonl.gz")
    
    print("\n2. Load and split:")
    print("""
from datasets import load_dataset

humaneval = load_dataset("openai_humaneval")
problems = [
    {
        'problem_id': item['task_id'],
        'prompt': item['prompt']
    }
    for item in humaneval['test']
]

# Use subset for training
train_problems = problems[:20]  # Start with 20 problems
    """)
    
    print("\n3. Generate preferences:")
    print("""
generator = PreferenceGenerator(api_key=API_KEY)
generator.generate_dataset(
    problems=train_problems,
    output_file='humaneval_preferences.json',
    num_responses=3,
    num_alternative_impls=3,
    num_test_cases=5
)
    """)


def estimate_costs():
    """Show cost estimates."""
    
    print("="*60)
    print("COST ESTIMATES")
    print("="*60)
    
    print("\nPer problem (3 responses, avg 3 functions per response):")
    print("  - Specifications: 9 calls")
    print("  - Test cases: 9 calls")
    print("  - Alternative impls: 27 calls")
    print("  - Total: ~45 Claude API calls")
    
    print("\nFor different dataset sizes:")
    sizes = [10, 50, 100, 200]
    for size in sizes:
        total_calls = size * 45
        est_cost_low = total_calls * 0.015  # ~$0.015 per call estimate
        est_cost_high = total_calls * 0.03
        print(f"  {size:3d} problems: {total_calls:5d} calls → ${est_cost_low:.0f}-${est_cost_high:.0f}")
    
    print("\nTraining cost (Modal/Colab Pro):")
    print("  - LoRA training: ~$1-2 per epoch")
    print("  - Full fine-tuning: ~$5-10 per epoch")
    
    print("\nTotal for 50 problem experiment:")
    print("  - Preference generation: $30-70")
    print("  - Training (LoRA, 3 epochs): $3-6")
    print("  - Evaluation: Free (local)")
    print("  - TOTAL: $33-76")
    
    print("="*60)


if __name__ == "__main__":
    import sys
    
    print("DPO Code Hallucination Reduction - Examples")
    print("="*60)
    print("\nChoose an example:")
    print("1. Minimal run (2 problems, test the pipeline)")
    print("2. HumanEval setup instructions")
    print("3. Cost estimates")
    print()
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        example_minimal_run()
    elif choice == "2":
        example_with_humaneval()
    elif choice == "3":
        estimate_costs()
    else:
        print("Invalid choice")
