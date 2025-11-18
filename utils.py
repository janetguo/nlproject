"""Utilities for code parsing and execution with ground truth tests."""

import ast
import sys
import io
import traceback
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import timeout_decorator


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


def calculate_test_pass_rate(code: str, test_cases: List[Dict]) -> Tuple[float, Dict[str, int]]:
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
    if not test_cases:
        return 0.0, {}
    
    passed = 0
    error_counts = {error_type: 0 for error_type in ErrorType}
    
    for test in test_cases:
        test_input = test.get('input')
        expected_output = test.get('output')
        
        error_type, actual_output = execute_code_with_test(code, test_input, expected_output)
        error_counts[error_type] += 1
        
        if error_type == ErrorType.SUCCESS:
            passed += 1
    
    pass_rate = passed / len(test_cases)
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
