"""
Adapted from APPS/LiveCodeBench eval github.
"""

import os
import sys

sys.set_int_max_str_digits(50000)

import json
import multiprocessing
from typing import Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from evaluation.testing_util import run_test
from evaluation.code_generation import CodeGenerationProblem


def _temp_run(sample, generation, debug, result, timeout):
    """
    Run test function that executes generated code and grades across unit tests.
    """
    try:
        if debug:
            print(f"Running test for problem: {sample}")
        
        # Result is list of test-cases and their output, metadata is just runtime
        res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
        result.append(res)

        if debug:
            print(f"Test completed with result: {result}")
    except Exception as e:
        if debug:
            print(f"Error in _temp_run: {e}")


def check_correctness(sample, generation, timeout=6, debug=False):
    """
    Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`
    """

    # Check for sample type (dataset vs CodeGenerationProblem)
    if isinstance(sample, CodeGenerationProblem):
        sample = sample.get_evaluation_sample()

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, timeout),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    if debug:
        print(f"Final result: {result}")
    return result[0]

def eval_sample(sample, generations, timeout=6, debug=False):
    """
    Evaluates a single sample across all code generations
    """
    results = []

    # Iterate over the generations for this sample
    for generation_idx, generation in enumerate(generations):  
        if debug:
            print(f"\nTesting solution {generation_idx}, {generation=}")

        # Default error state
        curr_res = [-2]
        try:

            # Check correctness and evaluate it
            curr_res = check_correctness(sample, generation=generation, timeout=timeout, debug=debug)
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed

            if not np.all(curr_res) and debug:
                print(f"Results were not all True: {curr_res}")

        except Exception as e:
            if debug:
                print(f"test framework exception = {repr(e)}{e}\n")

        finally:
            assert isinstance(curr_res, list)
            results.append(curr_res)

    return results


def eval_problems(samples, generations, timeout=6, debug=False, num_process_evaluate: int = 16, logging=False):
    """
    Evaluate multiple samples across their code generations.
    """

    #if samples.num_rows != len(generations):
        #raise ValueError("Mismatch: Number of samples must match number of generations lists.")

    # Evaluate all problems against their generations
    with tqdm(total=len(samples)) as pbar:
        with ProcessPoolExecutor(
            max_workers=1 if debug else num_process_evaluate
        ) as executor:
            futures = {
                executor.submit(eval_sample, sample, generations[index], timeout, debug): (index, sample.question_id if isinstance(sample, CodeGenerationProblem) else sample["problem_id"])
            for index, sample in enumerate(samples)
            }

            results = {}
            for future in as_completed(futures):
                index, problem_id = futures[future]
                results[(index, problem_id)] = future.result()
                pbar.update(1)
    
    if debug:
        print(f"\nHow to read results [-2] = compile error, [-1] = runtime error, [False] = failed test case, [True] = passed test case")
        #print(f"results = {res}")
    
    if logging:
        with open("evaluation_results.json", "w") as f:
            try:
                f.write(json.dumps(results))
            except Exception as e:
                import pdb; pdb.set_trace()
                print("didn't save problem due to {e}")

    return results



def compute_eval_results(results: Dict, debug=False):
    """
    Compute and output statistics for results from evaluating across samples

    Strict accuracies contains count of generations that pass all test-cases
    Per problem accuracies is percentage of generations passing all test-cases 
    """

    strict_accuracies = {} 
    per_problem_accuracies = {} 

    for (index, problem_id), generations in results.items():
        
        problem_results = np.array(generations, dtype=object)

        if problem_results.size == 0:
            strict_accuracies[(index, problem_id)] = 0
            per_problem_accuracies[(index, problem_id)] = 0.0
            continue 
        
        # Check for all the results for each generation if all test-cases pass
        passing_samples = np.array([np.all(np.array(subresult) > 0) for subresult in problem_results])

        strict_accuracies[(index, problem_id)] = np.sum(passing_samples)
        per_problem_accuracies[(index, problem_id)] = np.mean(passing_samples)

    # Compute and output overall statistics
    overall_strict_accuracy = np.mean([1 if count > 0 else 0 for count in strict_accuracies.values()])
    overall_per_problem_accuracy = np.mean(list(per_problem_accuracies.values()))

    if debug:
        print(f"Strict Accuracy (at least one passing sample per problem) = {overall_strict_accuracy:.3f}")
        print(f"Average Per-Problem Accuracy (fraction of passing samples per problem) = {overall_per_problem_accuracy:.3f}")

    return strict_accuracies, per_problem_accuracies
