"""
Sequential Refinement Sampling for Adaptive Code Generation

This module implements the sequential refinement approach for code generation.
In sequential refinement, a solution is refined through multiple iterations
with feedback from the environment between attempts.

Key features:
- Feedback-based refinement using verification results
- Multi-temperature configurations
- Solution verification against test cases
- Detailed logging and result tracking
"""

import json
import logging
import time
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from io import StringIO

# Import for model interfaces
from litellm import completion
from llm.methods import extract_code


class SequentialRefinementSampler:
    """
    Implements sequential refinement sampling for code generation.
    
    This approach iteratively improves a solution by providing feedback
    from verification between refinement attempts.
    """
    
    def __init__(
        self, 
        api_key: str, 
        temperatures: List[float],
        max_attempts: int,
        experiment_name: str = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize the sequential refinement sampler.
        
        Args:
            api_key: API key for the LLM service
            temperatures: List of temperatures to test
            max_attempts: Maximum number of refinement attempts
            experiment_name: Name for the experiment (for logging)
            model: Model to use for generation
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI Python library is required for this sampler")
            
        self.model = model
        self.api_key = api_key
        
        # Create experiment configurations
        self.seq_configs = [
            {"temperature": temp, "max_attempts": max_attempts}
            for temp in temperatures
        ]
        
        # Set up experiment tracking
        self.experiment_name = experiment_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(f"experiments/{self.experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self._setup_logging()
        
        # Create CSV file for detailed results
        self.results_file = self.experiment_dir / "results.csv"
        self._setup_results_file()
        
    def _setup_logging(self):
        """Set up logging configuration."""
        log_file = self.experiment_dir / "solver.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_results_file(self):
        """Create and initialize the CSV results file."""
        headers = [
            'timestamp',
            'problem_id',
            'temperature',
            'success',
            'attempts',
            'execution_time'
        ]
        
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def verifier(self, solution: str, problem: Dict) -> Tuple[bool, str]:
        """
        Verification function. Returns (success, error_message)
        
        Args:
            solution: The solution code string
            problem: The problem data with test cases
            
        Returns:
            A tuple of (success, error_message)
        """
        from solution_verifier import verify_solution_string
        
        # Use the solution_verifier module
        return verify_solution_string(solution, problem)

    def generate_solution(self, problem: Dict, previous_attempt: Optional[str] = None, 
                        error_message: Optional[str] = None, temperature: float = 0.7) -> Tuple[str, float]:
        """
        Generate a solution using the LLM with feedback from previous attempts.
        
        Args:
            problem: The problem data
            previous_attempt: The previous solution attempt (if any)
            error_message: Error message from previous attempt verification
            temperature: Temperature for generation
            
        Returns:
            A tuple of (solution_text, execution_time)
        """
        start_time = time.time()
        try:
            messages = []
            
            # System message
            messages.append({"role": "system", "content": "You are an expert Python programmer."})
            
            # Initial prompt
            starter_code = problem.get("starter_code", "")
            starter_code_section = f"\nHere is the starter code that should be included:\n```python\n{starter_code}\n```\n" if starter_code else ""
            
            base_prompt = f"""Given the following programming problem, provide a Python solution:
            {problem["question"]}
            {starter_code_section}
            Please provide only the code solution without any explanations.
            Your solution must be wrapped between ```python and ```."""
            
            messages.append({"role": "user", "content": base_prompt})
            
            # Add previous attempt and error if they exist
            if previous_attempt and error_message:
                # Include starter code if available
                starter_code = problem.get("starter_code", "")
                starter_code_section = f"\nHere is the starter code that should be included:\n```python\n{starter_code}\n```\n" if starter_code else ""
                
                feedback_prompt = f"""Your previous solution failed:
                
                ```python
                {extract_code(previous_attempt)}
                ```
                
                Error details:
                {error_message}
                {starter_code_section}
                Please provide a corrected solution that addresses these errors and passes the test case.
                Focus on fixing the specific failure and make sure your solution handles the test case properly.
                Remember to wrap your solution between ```python and ```."""
                messages.append({"role": "assistant", "content": previous_attempt})
                messages.append({"role": "user", "content": feedback_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            
            execution_time = time.time() - start_time
            response_text = response.choices[0].message.content
            
            return response_text, execution_time
            
        except Exception as e:
            self.logger.error(f"Error generating solution: {str(e)}")
            execution_time = time.time() - start_time
            return "", execution_time

    def _log_result(self, **kwargs):
        """
        Log results to the CSV file.
        
        Args:
            **kwargs: The values to log
        """
        row = [
            datetime.datetime.now().isoformat(),
            kwargs['problem_id'],
            kwargs['temperature'],
            kwargs['success'],
            kwargs['attempts'],
            kwargs['execution_time']
        ]
        
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def solve_problem_sequential(self, problem_id: int, problem: Dict, seq_config: Dict) -> Dict:
        """
        Solve a problem using sequential sampling approach.
        
        Args:
            problem_id: The ID of the problem
            problem: The problem data
            seq_config: Configuration for this sequential run
            
        Returns:
            A dictionary with the results
        """
        self.logger.info(f"Starting to solve problem {problem_id} sequentially with temperature {seq_config['temperature']}")
        
        temperature = seq_config["temperature"]
        max_attempts = seq_config["max_attempts"]
        
        total_time = 0
        success = False
        attempts = 0
        solution = None
        error = None
        all_attempts = []
        
        # Try solving the problem sequentially up to max_attempts
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            
            # Generate solution (possibly using feedback from previous attempt)
            solution, exec_time = self.generate_solution(
                problem, 
                previous_attempt=solution,
                error_message=error,
                temperature=temperature
            )
            total_time += exec_time
            
            attempt_record = {
                "attempt_number": attempt,
                "solution": solution,
                "execution_time": exec_time,
                "previous_error": error if attempt > 1 else None
            }
            
            # Verify the solution
            is_valid, error = self.verifier(solution, problem)
            
            if is_valid:
                success = True
                self.logger.info(f"Problem {problem_id} solved after {attempt} attempts")
                # Add the successful attempt to the record
                attempt_record["success"] = True
                attempt_record["error"] = None
                all_attempts.append(attempt_record)
                break
            
            # Enhance error with test case info if available
            if "Test Case" in error and ("Input:" in error or "Expected:" in error):
                # The error already contains test case details from verifier
                enhanced_error = error
            else:
                # Try to extract a test case from input_output
                input_output = problem.get('input_output', '{}')
                try:
                    io_data = json.loads(input_output)
                    if "inputs" in io_data and "outputs" in io_data and len(io_data["inputs"]) > 0:
                        test_idx = min(attempt - 1, len(io_data["inputs"]) - 1)  # Use different test case each time if possible
                        enhanced_error = f"{error}\n\nTest Case {test_idx + 1}:\nInput: {io_data['inputs'][test_idx]}\nExpected Output: {io_data['outputs'][test_idx]}"
                    elif "fn_name" in io_data and "inputs" in io_data and "outputs" in io_data and len(io_data["inputs"]) > 0:
                        test_idx = min(attempt - 1, len(io_data["inputs"]) - 1)
                        fn_name = io_data["fn_name"]
                        input_str = f"{fn_name}({', '.join(map(repr, io_data['inputs'][test_idx]))})"
                        enhanced_error = f"{error}\n\nTest Case {test_idx + 1}:\nFunction Call: {input_str}\nExpected Output: {io_data['outputs'][test_idx]}"
                    else:
                        enhanced_error = error
                except Exception as e:
                    self.logger.error(f"Error enhancing error message: {e}")
                    enhanced_error = error
            
            error = enhanced_error
            self.logger.info(f"Attempt {attempt} failed with error: {error}")
            
            # Add the verification result to the attempt record
            attempt_record["success"] = is_valid
            attempt_record["error"] = error
            all_attempts.append(attempt_record)
        
        # Log results to CSV
        self._log_result(
            problem_id=problem_id,
            temperature=temperature,
            success=success,
            attempts=attempts,
            execution_time=total_time
        )
        
        # Save detailed results for this problem
        problem_results = {
            'problem_id': problem_id,
            'success': success,
            'temperature': temperature,
            'attempts': attempts,
            'execution_time': total_time,
            'all_attempts': all_attempts,
            'final_solution': solution
        }
        
        with open(self.experiment_dir / f"problem_{problem_id}_details.json", 'w') as f:
            json.dump(problem_results, f, indent=2)
        
        return problem_results

    def solve_problem(self, problem_id: int, problem: Dict) -> List[Dict]:
        """
        Main solving function that tries all sequential configurations.
        
        Args:
            problem_id: The ID of the problem
            problem: The problem data
            
        Returns:
            A list of dictionaries with results for each configuration
        """
        self.logger.info(f"Starting to solve problem {problem_id}")
        
        results = []
        
        # Try each configuration
        for seq_config in self.seq_configs:
            result = self.solve_problem_sequential(problem_id, problem, seq_config)
            results.append(result)
        
        return results


def load_apps_dataset(file_name: str, start_idx: int = 200, end_idx: int = 220) -> Dict:
    """
    Load APPS dataset with a specific range of problems.
    
    Args:
        file_name: Path to the dataset file
        start_idx: Starting index for problems
        end_idx: Ending index for problems
        
    Returns:
        A dictionary of problems with problem_id as the key
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' library is required for loading datasets.")
        
    # Load dataset
    dataset = load_dataset('json', data_files=file_name, split='train')
    
    problem_indices = list(range(start_idx, end_idx))
    samples = dataset.select(problem_indices)
    
    problems = {}
    for sample in samples:
        problem_id = sample["problem_id"]
        problems[problem_id] = {
            "question": sample["question"],
            "input_output": sample["input_output"],
            "difficulty": sample["difficulty"],
            "starter_code": sample["starter_code"]
        }
    
    return problems


def main():
    """Main function to run the sequential refinement experiment"""
    # Configuration for sequential sampling
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
    max_attempts = 5
    
    # Initialize sampler with configurations
    # IMPORTANT: Replace with your own API key when using
    sampler = SequentialRefinementSampler(
        api_key="YOUR_API_KEY_HERE",
        temperatures=temperatures,
        max_attempts=max_attempts,
        experiment_name="sequential_refinement_experiment",
        model="gpt-4o-mini"  # Can be changed to other models
    )
    
    # Process problems - use a subset of the APPS dataset
    problems = load_apps_dataset("./data/apps_test_300.jsonl", 200, 210)
    all_results = []
    
    for problem_id, problem in problems.items():
        results = sampler.solve_problem(problem_id, problem)
        all_results.extend(results)
    
    # Save overall results
    with open(sampler.experiment_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Sequential refinement experiment complete. Results saved to {sampler.experiment_dir}")


if __name__ == "__main__":
    main()