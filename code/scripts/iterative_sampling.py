"""
Iterative Sampling Implementation for Adaptive Code Generation

This module implements the iterative sampling approach for code generation.
In iterative sampling, an initial solution is generated and then iteratively refined
at a given temperature, focusing on improving a single solution path.

Key features:
- Iterative refinement of generated solutions
- Multi-temperature configurations
- Solution verification against test cases
- Detailed logging and result tracking
"""

import json
import logging
import time
import csv
import datetime
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures
from threading import Lock
from dataclasses import dataclass

# Import necessary modules for model interfaces
from litellm import completion
from llm.methods import extract_code
from llm.prompts.code_generation import get_generic_question_template, get_iterative_refinement_template, PromptConstants
from evaluation.compute_code_generation_metrics import eval_lcb


@dataclass
class BranchConfig:
    """Configuration for a sampling branch"""
    temperature: float
    num_samples: int


@dataclass
class Branch:
    """Results from a sampling branch"""
    config: BranchConfig
    solutions: List[str]
    verified: bool = False
    execution_time: float = 0.0


class IterativeSampler:
    """
    Implements iterative refinement sampling strategy for code generation.
    
    An initial solution is generated and then iteratively refined at the specified temperature,
    focusing on improving a single solution path.
    """
    
    def __init__(
        self, 
        api_key: str, 
        temperatures: List[float],
        sample_counts: List[int],
        experiment_name: str = None,
        max_workers: int = 5,
        model: str = "gpt-4o-mini",
        max_iterations: int = 5
    ):
        """
        Initialize the iterative sampler.
        
        Args:
            api_key: API key for the LLM service
            temperatures: List of temperatures to test
            sample_counts: List of sample counts to test
            experiment_name: Name for the experiment (for logging)
            max_workers: Maximum number of parallel workers
            model: Model to use for generation
            max_iterations: Maximum number of iterative refinements per solution
        """
        self.api_key = api_key
        self.max_workers = max_workers
        self.model = model
        self.max_iterations = max_iterations
        
        # Create experiment configurations from all combinations
        self.branch_configs = [
            BranchConfig(temp, samples)
            for temp in temperatures
            for samples in sample_counts
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
        
        # Add lock for thread-safe file operations
        self.csv_lock = Lock()
        self.log_lock = Lock()

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
            'num_samples',
            'verified',
            'solutions_generated',
            'execution_time',
            'branch_success',
            'difficulty'
        ]
        
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def completion(self, messages, temperature):
        """
        Generate a completion using the LLM.
        
        Args:
            messages: The messages for the API call
            temperature: The temperature for generation
            
        Returns:
            The generated completion text
        """
        response = completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            api_key=self.api_key
        )

        return response["choices"][0]["message"]["content"]

    def verifier(self, solution: str, problem: Dict) -> bool:
        """
        Verification function to check if a solution is correct.
        
        Args:
            solution: The generated solution text
            problem: The problem data with input/output test cases
            
        Returns:
            True if the solution passes all tests, False otherwise
        """
        generated_code = extract_code(solution)
        if not generated_code:
            return False

        input_output = problem.get('input_output')
        if not input_output:
            with self.log_lock:
                self.logger.warning(f"No input/output data: {problem.get('problem_id', 'Unknown')}")
            return False
        
        result = eval_lcb(problem, generated_code)
        
        return result

    def verifier_multiple(self, solutions, problem: Dict) -> List[bool]:
        """
        Verify multiple solutions and return the output of each as list of bools.
        
        Args:
            solutions: List of solutions to verify
            problem: The problem data
            
        Returns:
            List of boolean results for each solution
        """
        results = []
        for solution in solutions:
            result = self.verifier(solution, problem)
            results.append(result)
        
        return results

    def generate_solution(self, problem: Dict, temperature: float) -> Tuple[str, float]:
        """
        Generate a single solution using the LLM API.
        
        Args:
            problem: The problem data
            temperature: The temperature for generation
            
        Returns:
            A tuple of (solution_text, execution_time)
        """
        start_time = time.time()
        try:
            # Generate prompt using template and problem
            prompt = get_generic_question_template(problem)
            messages = [
                {"content": PromptConstants.SYSTEM_MESSAGE_GENERIC, "role": "system"}, 
                {"content": prompt, "role": "user"}
            ]

            response = self.completion(messages=messages, temperature=temperature)
            
            execution_time = time.time() - start_time
            
            return response, execution_time
        
        except Exception as e:
            with self.log_lock:
                self.logger.error(f"Error generating solution: {str(e)}")
            execution_time = time.time() - start_time
            return "", execution_time

    def generate_solution_iterative(self, problem: Dict, temperature: float):
        """
        Iteratively generate and refine solutions.
        
        Args:
            problem: The problem data
            temperature: Base temperature for generation
            
        Returns:
            A tuple of (solutions_list, total_execution_time)
        """
        start_time = time.time()
        solutions = []

        for i in range(self.max_iterations):
            # For refinement iterations, we can vary the temperature slightly
            # for more exploration
            curr_temperature = temperature
            if i > 0:
                # Randomize temperature a bit for subsequent iterations
                curr_temperature = random.uniform(max(0.1, temperature - 0.2), 
                                               min(1.3, temperature + 0.2))

            if i == 0:
                # First solution is generated normally
                curr_solution, _ = self.generate_solution(problem, temperature=curr_temperature)
            else:
                # Subsequent solutions refine the previous one
                prompt = get_iterative_refinement_template(problem, solutions[-1])
                messages = [
                    {"content": PromptConstants.SYSTEM_MESSAGE_GENERIC, "role": "system"}, 
                    {"content": prompt, "role": "user"}
                ]
                curr_solution = self.completion(messages=messages, temperature=curr_temperature)
            
            solutions.append(curr_solution)

        execution_time = time.time() - start_time

        return solutions, execution_time

    def process_single_sample(self, problem: Dict, temperature: float) -> Tuple[Optional[List[bool]], float]:
        """
        Process a single sample branch for threading.
        
        Args:
            problem: The problem data
            temperature: The temperature for generation
            
        Returns:
            A tuple of (verification_results, execution_time) if any solution is verified, 
            or (None, execution_time) if none are verified
        """
        solutions, exec_time = self.generate_solution_iterative(problem, temperature)
        results = self.verifier_multiple(solutions, problem)
        if solutions and any(results):
            return results, exec_time
        return None, exec_time

    def process_branch(self, problem_id: int, problem: Dict, branch_config: BranchConfig) -> Branch:
        """
        Process a single branch with its configuration using multiple threads.
        
        Args:
            problem_id: The ID of the problem
            problem: The problem data
            branch_config: Configuration for this branch
            
        Returns:
            A Branch object with the results
        """
        solutions = []
        total_time = 0
        verified = False
        
        # Create a thread pool for parallel API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(
                    self.process_single_sample, 
                    problem, 
                    branch_config.temperature
                ): i for i in range(branch_config.num_samples)
            }
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_sample):
                solution, exec_time = future.result()
                total_time += exec_time
                
                if solution:
                    solutions.append(solution)
                    verified = True
                    # Cancel remaining futures if we found a verified solution
                    for f in future_to_sample:
                        if not f.done():
                            f.cancel()
                    break
                
        # Log results to CSV
        self._log_result(
            problem_id=problem_id,
            temperature=branch_config.temperature,
            num_samples=branch_config.num_samples,
            verified=verified,
            solutions_generated=str(solutions),
            execution_time=total_time,
            branch_success=verified,
            difficulty=problem.get("difficulty", "unknown")
        )
        
        return Branch(branch_config, str(solutions), verified=verified, execution_time=total_time)

    def _log_result(self, **kwargs):
        """
        Thread-safe logging to the CSV file.
        
        Args:
            **kwargs: The values to log
        """
        row = [
            datetime.datetime.now().isoformat(),
            kwargs['problem_id'],
            kwargs['temperature'],
            kwargs['num_samples'],
            kwargs['verified'],
            kwargs['solutions_generated'],
            kwargs['execution_time'],
            kwargs['branch_success'],
            kwargs['difficulty']
        ]
        
        with self.csv_lock:
            with open(self.results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

    def solve_problem(self, problem_id: int, problem: Dict) -> Dict:
        """
        Main solving function that tries all branch configurations in parallel.
        
        Args:
            problem_id: The ID of the problem
            problem: The problem data
            
        Returns:
            A dictionary with the results
        """
        with self.log_lock:
            self.logger.info(f"Starting to solve problem {problem_id}")
        
        results = []
        success = False
        
        # Process branch configurations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all branches
            future_to_branch = {
                executor.submit(
                    self.process_branch,
                    problem_id,
                    problem,
                    branch_config
                ): branch_config for branch_config in self.branch_configs
            }
            
            # Process completed branches as they finish
            for future in concurrent.futures.as_completed(future_to_branch):
                branch_config = future_to_branch[future]
                branch = future.result()
                
                results.append({
                    'temperature': branch_config.temperature,
                    'num_samples': branch_config.num_samples,
                    'verified': branch.verified,
                    'solutions_count': branch.solutions,
                    'execution_time': branch.execution_time
                })
                
                if branch.verified:
                    success = True
                    with self.log_lock:
                        self.logger.info("Found successful solution, continuing to try other configurations")
        
        # Save detailed results for this problem
        problem_results = {
            'problem_id': problem_id,
            'success': success,
            'branches': results
        }
        
        with open(self.experiment_dir / f"problem_{problem_id}_details.json", 'w') as f:
            json.dump(problem_results, f, indent=2)
        
        return problem_results


def load_apps_dataset(file_name: str, start_idx: int = 0, end_idx: int = 300) -> Dict:
    """
    Load Apps dataset as a dictionary with problem_id as the key.
    
    Args:
        file_name: The path to the dataset file
        start_idx: The starting index to load
        end_idx: The ending index to load
        
    Returns:
        A dictionary of problems with problem_id as the key
    """
    from datasets import load_dataset
    
    # Load dataset
    dataset = load_dataset('json', data_files=file_name, split='train')
    samples = dataset.select(range(start_idx, end_idx, 1))

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


def load_lcb_dataset(count: int) -> Dict:
    """
    Load Live Code Bench dataset as a dictionary with problem_id as the key.
    
    Args:
        count: The number of problems to load per difficulty level
        
    Returns:
        A dictionary of problems with problem_id as the key
    """
    from evaluation.code_generation import load_code_generation_dataset, Difficulty
    
    dataset = load_code_generation_dataset(release_version="release_v5", cutoff_date="2024-05-01")
    easy_problems = [problem for problem in dataset if problem.difficulty == Difficulty.EASY][:count]
    medium_problems = [problem for problem in dataset if problem.difficulty == Difficulty.MEDIUM][:count]
    hard_problems = [problem for problem in dataset if problem.difficulty == Difficulty.HARD][:count]

    print(f"Number of easy problems: {len(easy_problems)}")
    print(f"Number of medium problems: {len(medium_problems)}")
    print(f"Number of hard problems: {len(hard_problems)}")

    samples = easy_problems + medium_problems + hard_problems

    problems = {}
    for sample in samples:
        problem_id = sample.question_id
        problems[problem_id] = {
            "question": sample.question_content,
            "input_output": sample.get_evaluation_sample()["input_output"],
            "difficulty": sample.difficulty,
            "starter_code": sample.starter_code
        }
    return problems


def main():
    """Main function to run the iterative sampling experiment"""
    # Configuration
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
    sample_counts = [1, 5, 10]
    
    # Initialize sampler with configurations
    # IMPORTANT: Replace with your own API key when using
    sampler = IterativeSampler(
        api_key="YOUR_API_KEY_HERE",
        temperatures=temperatures,
        sample_counts=sample_counts,
        experiment_name="iterative_sampling_experiment",
        max_workers=5,
        model="gpt-4o-mini",  # Can be changed to other models
        max_iterations=5
    )

    # Process problems 
    problems = load_lcb_dataset(10)
    # Alternative: Use APPS dataset
    # problems = load_apps_dataset("./data/apps_test_300.jsonl", 0, 10)
    
    results = []
    for problem_id, problem in problems.items():
        result = sampler.solve_problem(problem_id, problem)
        results.append(result)
    
    # Save overall results
    with open(sampler.experiment_dir / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()