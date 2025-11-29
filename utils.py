"""Utilities for code parsing and execution with ground truth tests."""

import ast
import sys
import io
import traceback
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import timeout_decorator
from instrument_test_code import instrument_test_code
from datasets import load_dataset
import inspect


class ErrorType(Enum):
    """Categories of code execution errors."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    WRONG_OUTPUT = "wrong_output"
    SUCCESS = "success"


@timeout_decorator.timeout(5)
def execute_code_with_test(code: str, test_input: Any, expected_output: Any) -> Tuple[ErrorType, Optional[Any]]:
    """
    Execute code with a single test case.
    
    Args:
        code: Generated code (complete solution)
        test_input: Input for the test case
        expected_output: Expected output
    
    Returns:
        (error_type, actual_output)
    """
    # Create isolated namespace
    namespace = {}
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        # Try to execute the code
        exec(code, namespace)
        
        # Find the main function (entry point)
        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith('_'):
                func = obj
                break
        
        if func is None:
            sys.stdout = old_stdout
            return ErrorType.RUNTIME_ERROR, None
        
        # Execute with test input
        if isinstance(test_input, (list, tuple)):
            result = func(*test_input)
        else:
            result = func(test_input)
        
        sys.stdout = old_stdout
        
        # Check if output matches expected
        if result == expected_output:
            return ErrorType.SUCCESS, result
        else:
            return ErrorType.WRONG_OUTPUT, result
            
    except SyntaxError:
        sys.stdout = old_stdout
        return ErrorType.SYNTAX_ERROR, None
    except timeout_decorator.TimeoutError:
        sys.stdout = old_stdout
        return ErrorType.TIMEOUT, None
    except Exception as e:
        sys.stdout = old_stdout
        return ErrorType.RUNTIME_ERROR, None

@timeout_decorator.timeout(5)
def calculate_test_pass_rate(code: str, test_code: str) -> Tuple[float, Dict[str, int]]:
    """
    Calculate pass rate on ground truth test cases.
    
    Args:
        code: Generated code
        test_cases: List of dicts with 'input' and 'output' keys
    
    Returns:
        (pass_rate, error_counts):
            - pass_rate: fraction of tests passed (0.0 to 1.0)
            - error_counts: dict mapping ErrorType to count
    """

    # 1. Prepare execution environment for candidate code
    candidate_env = {}
    try:
        exec(code, candidate_env)
    except timeout_decorator.TimeoutError:
        return 0.0, {
            ErrorType.TIMEOUT: 1
        }
    except Exception as e:
        # Candidate code failed to compile or execute
        return 0.0, {
            ErrorType.RUNTIME_ERROR: 1 
        }

    # Candidate solution must define the expected entry point
    candidate = None
    for v in candidate_env.values():
        if inspect.isfunction(v):
            candidate = v
            break
    if candidate is None:
        return 0.0, {
            ErrorType.RUNTIME_ERROR: 1 
        }

    # 2. Instrument the HumanEval test code
    env, results = instrument_test_code(test_code)

    # 3. Run the instrumented check(candidate)
    try:
        check_fn = env["check"]
        check_fn(candidate)
    except timeout_decorator.TimeoutError:
        return 0.0, {
            ErrorType.TIMEOUT: 1
        }
    except Exception as e:
        print("EXCEPTION", e)
        # Unexpected crash during tests, not an assertion failure
        # The assertion transformer catches normal assertion failures.
        return 0.0, {
            ErrorType.RUNTIME_ERROR: 1, 
            ErrorType.SUCCESS: results["passed"],
            ErrorType.WRONG_OUTPUT: results["failed"]
        }

    # 4. Compute pass rate
    total_asserts = results["passed"] + results["failed"]
    pass_rate = (results["passed"] / total_asserts) if total_asserts > 0 else 0.0

    error_counts = {
        ErrorType.SUCCESS: results["passed"],
        ErrorType.WRONG_OUTPUT: results["failed"]
    }

    return pass_rate, error_counts


def validate_code_syntax(code: str) -> bool:
    """Check if code has valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def get_error_summary(error_counts: Dict[ErrorType, int]) -> Dict[str, Any]:
    """
    Generate summary statistics from error counts.
    
    Returns dict with:
        - total_tests: int
        - passed: int
        - syntax_errors: int
        - runtime_errors: int
        - timeouts: int
        - wrong_outputs: int
        - compile_rate: float (% that parsed)
    """
    total = sum(error_counts.values())
    
    return {
        'total_tests': total,
        'passed': error_counts.get(ErrorType.SUCCESS, 0),
        'syntax_errors': error_counts.get(ErrorType.SYNTAX_ERROR, 0),
        'runtime_errors': error_counts.get(ErrorType.RUNTIME_ERROR, 0),
        'timeouts': error_counts.get(ErrorType.TIMEOUT, 0),
        'wrong_outputs': error_counts.get(ErrorType.WRONG_OUTPUT, 0),
        'compile_rate': 1.0 - (error_counts.get(ErrorType.SYNTAX_ERROR, 0) / total) if total > 0 else 0.0
    }

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
    
    problems.sort(key=lambda x: x["problem_id"])

    return problems

def split_dataset(
    problems: List[Dict],
    train_ratio: float = 0.2,
    test_ratio: float = 0.3,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
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
    test_end = train_end + int(n * test_ratio)
    
    train = shuffled[:train_end]
    test = shuffled[train_end:test_end]
    
    print(f"\nDataset split (seed={seed}):")
    print(f"  Train: {len(train)} problems ({train_ratio*100:.0f}%)")
    print(f"  Test:  {len(test)} problems ({test_ratio*100:.0f}%)")
    
    return train, test