"""
Solution Verifier for Code Generation

This module provides utilities for verifying code solutions against test cases.
It ensures generated solutions meet functional requirements by running them
against provided inputs and comparing outputs.

Key features:
- Safe execution of Python code in a controlled environment
- Support for various input/output formats
- Timeout handling to prevent infinite loops
"""

import json
import subprocess
import sys
import signal
import io
from typing import Dict, Tuple


def extract_code(solution: str) -> str:
    """
    Extract Python code from a markdown-formatted solution.
    
    Args:
        solution: The solution text, potentially containing markdown
        
    Returns:
        The extracted Python code
    """
    import re
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, solution, re.DOTALL)

    if match:
        extracted_text = match.group(1)
        return extracted_text
    
    # If no Python block found, check for any code block
    pattern = r"```\s*(.*?)\s*```"
    match = re.search(pattern, solution, re.DOTALL)
    if match:
        extracted_text = match.group(1)
        return extracted_text
        
    # Return the original text if no code blocks found
    return solution


def run_and_check(python_code_string: str, input_output_string: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Runs Python code, checks against test cases, and handles infinite loops.

    Args:
        python_code_string: The Python code as a string.
        input_output_string: JSON string with "inputs" and "outputs".
        timeout: Timeout in seconds (default: 5).
        
    Returns:
        A tuple of (success, error_message)
    """
    try:
        input_output_data = json.loads(input_output_string)
        test_cases = {"inputs": [], "outputs": []}

        # Handle different input/output formats
        if "inputs" in input_output_data and "outputs" in input_output_data:
            test_cases["inputs"] = [str(inp) + "\n" for inp in input_output_data["inputs"]]
            test_cases["outputs"] = [str(out) + "\n" for out in input_output_data["outputs"]]
        elif "fn_name" in input_output_data:
            fn_name = input_output_data["fn_name"]
            for inp, out in zip(input_output_data["inputs"], input_output_data["outputs"]):
                input_str = f"{fn_name}({', '.join(map(repr, inp))})\n"
                test_cases["inputs"].append(input_str)
                test_cases["outputs"].append(str(out) + "\n")
        else:
            return False, "Error: Invalid input_output format."

        inputs = test_cases["inputs"]
        expected_outputs = test_cases["outputs"]

        if len(inputs) != len(expected_outputs):
            return False, "Error: Number of inputs/outputs mismatch."

        # Run each test case
        for i in range(len(inputs)):
            try:
                process = subprocess.run(
                    ["python", "-c", python_code_string],
                    input=inputs[i].encode('utf-8'),
                    capture_output=True,
                    check=False,  # Don't raise an exception
                    timeout=timeout  # Set the timeout
                )
            except subprocess.TimeoutExpired:
                error_msg = f"Test Case {i + 1}: Timed out (infinite loop suspected)."
                return False, error_msg

            if process.returncode != 0:
                error_msg = f"Runtime Error in test case {i + 1}:\n{process.stderr.decode('utf-8')}"
                return False, error_msg

            actual_output = process.stdout.decode('utf-8')
            if actual_output.strip() != expected_outputs[i].strip():
                error_msg = (
                    f"Test Case {i + 1} Failed:\n"
                    f"  Input:    {inputs[i].strip()}\n"
                    f"  Expected: {expected_outputs[i].strip()}\n"
                    f"  Actual:   {actual_output.strip()}"
                )
                return False, error_msg

        return True, ""

    except json.JSONDecodeError:
        return False, "Error: Invalid JSON format."
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"


def verify_solution_file(solution_path: str, problem_data: Dict) -> Tuple[bool, str]:
    """
    Verify a solution from a file against a problem's test cases.
    
    Args:
        solution_path: Path to the solution Python file
        problem_data: Dictionary containing problem data with input_output
        
    Returns:
        A tuple of (success, error_message)
    """
    try:
        with open(solution_path, 'r') as f:
            code = f.read()
        
        input_output = problem_data.get('input_output')
        if not input_output:
            return False, "No input/output data found in problem"
            
        return run_and_check(code, input_output)
    
    except Exception as e:
        return False, f"Error verifying solution: {str(e)}"


def verify_solution_string(solution: str, problem_data: Dict) -> Tuple[bool, str]:
    """
    Verify a solution from a string against a problem's test cases.
    
    Args:
        solution: Solution string, potentially with markdown formatting
        problem_data: Dictionary containing problem data with input_output
        
    Returns:
        A tuple of (success, error_message)
    """
    code = extract_code(solution)
    if not code:
        return False, "No valid code found in solution"
    
    input_output = problem_data.get('input_output')
    if not input_output:
        return False, "No input/output data found in problem"
        
    return run_and_check(code, input_output)


def evaluate_solutions_batch(solutions: Dict[str, str], problem_data: Dict) -> Dict[str, Tuple[bool, str]]:
    """
    Evaluate multiple solutions for a single problem.
    
    Args:
        solutions: Dictionary mapping solution IDs to solution strings
        problem_data: Dictionary containing problem data with input_output
        
    Returns:
        Dictionary mapping solution IDs to (success, error_message) tuples
    """
    results = {}
    
    for solution_id, solution in solutions.items():
        success, error_msg = verify_solution_string(solution, problem_data)
        results[solution_id] = (success, error_msg)
        
    return results


def main():
    """Example usage of the verifier"""
    # Example problem and solution
    problem_data = {
        "input_output": json.dumps({
            "inputs": ["5", "10"],
            "outputs": ["15", "20"]
        })
    }
    
    # A correct solution
    correct_solution = """
    ```python
    a = int(input())
    print(a + 10)
    ```
    """
    
    # An incorrect solution
    incorrect_solution = """
    ```python
    a = int(input())
    print(a + 5)  # Wrong implementation
    ```
    """
    
    # Infinite loop solution
    loop_solution = """
    ```python
    while True:
        pass
    ```
    """
    
    # Test the verifier
    print("Testing correct solution:")
    success, msg = verify_solution_string(correct_solution, problem_data)
    print(f"Success: {success}, Message: {msg}")
    
    print("\nTesting incorrect solution:")
    success, msg = verify_solution_string(incorrect_solution, problem_data)
    print(f"Success: {success}, Message: {msg}")
    
    print("\nTesting infinite loop solution:")
    success, msg = verify_solution_string(loop_solution, problem_data)
    print(f"Success: {success}, Message: {msg}")


if __name__ == "__main__":
    main()