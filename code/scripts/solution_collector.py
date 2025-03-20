"""
Solution Collector for Code Generation Analysis

This module handles collecting solutions at different temperatures from LLMs
for analysis, without performing verification. Used for qualitative and quantitative
analysis of solution diversity and characteristics.

Key features:
- Multi-temperature solution collection
- Detailed logging and organization of solutions
- Compatible with different LLM providers
"""

import json
import logging
import time
import csv
import os
from pathlib import Path
import datetime
from typing import Dict, List, Any

# Import for extracting code from responses
from llm.methods import extract_code


class MultiTempSolutionCollector:
    """
    Collects and stores solutions at multiple temperatures without verification.
    Used for analysis of solution diversity and characteristics.
    """
    
    def __init__(
        self, 
        api_key: str,
        temperatures: List[float],
        experiment_name: str = None,
        model: str = "gpt-4o"
    ):
        """
        Initialize the solution collector.
        
        Args:
            api_key: API key for the LLM service
            temperatures: List of temperatures to collect solutions at
            experiment_name: Name for the experiment (for logging)
            model: Model to use for generation
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            self.logger.error("OpenAI Python library is required for this collector")
            raise
            
        self.temperatures = temperatures
        self.model = model
        
        # Set up experiment tracking
        self.experiment_name = experiment_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"experiments/{self.experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories for each temperature
        self.temp_dirs = {}
        for temp in temperatures:
            temp_dir = self.output_dir / f"temp_{temp}"
            temp_dir.mkdir(exist_ok=True)
            self.temp_dirs[temp] = temp_dir
        
        # Configure logging
        self._setup_logging()
        
        # Create CSV file for tracking
        self.results_file = self.output_dir / "collection_results.csv"
        self._setup_results_file()
        
    def _setup_logging(self):
        """Set up logging configuration."""
        log_file = self.output_dir / "collector.log"
        
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
            'execution_time',
            'solution_length',
            'status'
        ]
        
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def generate_solution(self, problem: Dict, temperature: float) -> tuple:
        """
        Generate a solution using the LLM with a single prompt.
        
        Args:
            problem: The problem data
            temperature: The temperature for generation
            
        Returns:
            A tuple of (solution_text, execution_time, status)
        """
        start_time = time.time()
        try:
            # Include starter code if available
            starter_code = problem.get("starter_code", "")
            starter_code_section = f"\nHere is the starter code that should be included:\n```python\n{starter_code}\n```\n" if starter_code else ""
            
            prompt = f"""Given the following programming problem, provide a Python solution:
            {problem["question"]}
            {starter_code_section}
            Please provide only the code solution without any explanations.
            Your solution must be wrapped between ```python and ```."""
            
            messages = [
                {"role": "system", "content": "You are an expert Python programmer."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            
            execution_time = time.time() - start_time
            response_text = response.choices[0].message.content
            
            return response_text, execution_time, "success"
            
        except Exception as e:
            self.logger.error(f"Error generating solution: {str(e)}")
            execution_time = time.time() - start_time
            return "", execution_time, f"error: {str(e)}"

    def _log_result(self, problem_id: int, temperature: float, execution_time: float, solution: str, status: str):
        """
        Log results to the CSV file.
        
        Args:
            problem_id: The ID of the problem
            temperature: The temperature used for generation
            execution_time: Time taken to generate the solution
            solution: The generated solution text
            status: Status of the generation (success/error)
        """
        row = [
            datetime.datetime.now().isoformat(),
            problem_id,
            temperature,
            execution_time,
            len(solution),
            status
        ]
        
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def collect_solution(self, problem_id: int, problem: Dict) -> Dict[float, Dict[str, Any]]:
        """
        Collect solutions for a problem at different temperatures without running them.
        
        Args:
            problem_id: The ID of the problem
            problem: The problem data
            
        Returns:
            A dictionary mapping temperatures to solution results
        """
        self.logger.info(f"Collecting solutions for problem {problem_id} at multiple temperatures")
        
        results = {}
        
        # Generate solutions at each temperature
        for temp in self.temperatures:
            self.logger.info(f"Generating solution for problem {problem_id} at temperature {temp}")
            
            # Get the solution at this temperature
            solution, exec_time, status = self.generate_solution(problem, temp)
            
            # Extract code
            code = extract_code(solution)
            
            # Get the directory for this temperature
            temp_dir = self.temp_dirs[temp]
            
            # Save the code file
            solution_file = temp_dir / f"problem_{problem_id}_solution.py"
            with open(solution_file, 'w') as f:
                f.write(code)
            
            # Create the result entry
            result = {
                'problem_id': problem_id,
                'temperature': temp,
                'execution_time': exec_time,
                'status': status,
                'full_response': solution,
                'extracted_code': code,
                'question': problem.get('question', '')
            }
            
            # Save detailed result for this temperature
            with open(temp_dir / f"problem_{problem_id}_details.json", 'w') as f:
                json.dump(result, f, indent=2)
            
            # Log to CSV
            self._log_result(problem_id, temp, exec_time, code, status)
            
            # Store in overall results
            results[temp] = result
            
            # Add a small delay between API calls to avoid rate limits
            time.sleep(1)
        
        return results


def load_apps_dataset(file_name: str, start_idx: int = 0, end_idx: int = 300) -> Dict:
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
        print("The 'datasets' library is required for loading datasets.")
        raise
        
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
    """Main function to run the solution collection experiment"""
    # Define temperatures to test
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.3]
    
    # Initialize collector
    # IMPORTANT: Replace with your own API key when using
    collector = MultiTempSolutionCollector(
        api_key="YOUR_API_KEY_HERE",
        temperatures=temperatures,
        experiment_name="solution_collection_experiment",
        model="gpt-4o"
    )
    
    # Load problems - adjust range as needed
    problems = load_apps_dataset("./data/apps_test_300.jsonl", 200, 210)
    
    # Data structure to track results
    all_results = {temp: [] for temp in temperatures}
    problem_ids = []
    
    # Collect solutions for each problem at each temperature
    for problem_id, problem in problems.items():
        problem_ids.append(problem_id)
        
        # This will generate solutions at all temperatures
        results = collector.collect_solution(problem_id, problem)
        
        # Organize results by temperature
        for temp, result in results.items():
            all_results[temp].append(result)
        
        # Log progress
        print(f"Completed problem {problem_id} at all temperatures")
    
    # Generate summary statistics for each temperature
    summary_stats = {}
    for temp in temperatures:
        temp_results = all_results[temp]
        avg_time = sum(r['execution_time'] for r in temp_results) / len(temp_results)
        avg_length = sum(len(r['extracted_code']) for r in temp_results) / len(temp_results)
        success_count = sum(1 for r in temp_results if r['status'] == 'success')
        
        summary_stats[str(temp)] = {
            "average_execution_time": avg_time,
            "average_solution_length": avg_length,
            "success_rate": success_count / len(temp_results),
            "total_problems": len(temp_results)
        }
    
    # Save summary of all results
    with open(collector.output_dir / "collection_summary.json", 'w') as f:
        json.dump({
            "total_problems": len(problems),
            "temperatures_tested": temperatures,
            "timestamp": datetime.datetime.now().isoformat(),
            "problem_ids": problem_ids,
            "summary_by_temperature": summary_stats
        }, f, indent=2)
    
    print(f"Collection complete. Solutions saved to {collector.output_dir}")


if __name__ == "__main__":
    main()