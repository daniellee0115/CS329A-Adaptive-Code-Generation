"""
Data Analysis for Adaptive Code Generation

This module provides utilities for analyzing the results of sampling experiments.
It includes functions for generating graphs and computing statistics over various
sampling strategies, temperatures, and problem types.

Key features:
- Performance metric visualization across temperature settings
- Comparison of different sampling strategies
- Analysis of solution diversity and characteristics
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_experiment_results(experiment_dir: str) -> Dict:
    """
    Load all results from an experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Dictionary containing parsed experiment results
    """
    experiment_path = Path(experiment_dir)
    
    # Look for summary files first
    summary_file = experiment_path / "collection_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    
    # If no summary file, try to load all problem detail files
    all_results = {"problems": {}}
    
    for file in experiment_path.glob("problem_*_details.json"):
        with open(file, 'r') as f:
            problem_data = json.load(f)
            problem_id = problem_data.get("problem_id")
            if problem_id:
                all_results["problems"][problem_id] = problem_data
    
    return all_results


def plot_temperature_metrics(data: Dict, output_file: Optional[str] = "temperature_metrics.png"):
    """
    Plot performance metrics by temperature.
    
    Args:
        data: Dictionary containing summary data by temperature
        output_file: Path to save the output figure
    """
    if "summary_by_temperature" in data:
        summary_data = data["summary_by_temperature"]
    else:
        summary_data = data  # Assume data is already the summary dict
    
    # Extract data for plotting
    temperatures = list(summary_data.keys())
    execution_times = [summary_data[temp]["average_execution_time"] for temp in temperatures]
    solution_lengths = [summary_data[temp]["average_solution_length"] for temp in temperatures]
    
    # Define colors for temperature levels
    temperature_colors = {
        "0.1": "#636EFA",  # Blue
        "0.3": "#2CA02C",  # Green
        "0.5": "#FF7F0E",  # Orange
        "0.7": "#8C564B",  # Brown
        "0.9": "#D62728",  # Red
        "1.3": "#9467BD",  # Purple
    }
    
    # Set plot style
    plt.rcParams['figure.dpi'] = 100  
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set the width of the bars
    bar_width = 0.6
    
    # Plot execution time
    for i, temp in enumerate(temperatures):
        ax1.bar(
            i,
            summary_data[temp]["average_execution_time"],
            width=bar_width,
            color=temperature_colors.get(temp, "#7F7F7F"),
            label=f"Temperature {temp}"
        )
    
    # Plot solution length
    for i, temp in enumerate(temperatures):
        ax2.bar(
            i,
            summary_data[temp]["average_solution_length"],
            width=bar_width,
            color=temperature_colors.get(temp, "#7F7F7F")
        )
    
    # Add titles and labels
    ax1.set_title('Average Execution Time by Temperature', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Temperature', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_xticks(range(len(temperatures)))
    ax1.set_xticklabels(temperatures)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2.set_title('Average Solution Length by Temperature', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Temperature', fontsize=12) 
    ax2.set_ylabel('Solution Length (characters)', fontsize=12)
    ax2.set_xticks(range(len(temperatures)))
    ax2.set_xticklabels(temperatures)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(execution_times):
        ax1.text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')
    
    for i, v in enumerate(solution_lengths):
        ax2.text(i, v + 10, f"{int(v)}", ha='center', fontweight='bold')
    
    # Add a legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=5)
    
    # Improve layout and spacing
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.suptitle('Performance Metrics by Temperature', fontsize=16, fontweight='bold')
    
    # Save the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Graph has been saved as '{output_file}'")


def plot_strategy_comparison(parallel_data: Dict, serial_data: Dict, iterative_data: Dict, 
                            output_file: Optional[str] = "strategy_comparison.png"):
    """
    Plot comparison of different sampling strategies.
    
    Args:
        parallel_data: Results from parallel sampling
        serial_data: Results from serial sampling
        iterative_data: Results from iterative sampling
        output_file: Path to save the output figure
    """
    # Extract success rates for each strategy by temperature
    strategies = ["Parallel", "Serial", "Iterative"]
    temperatures = ["0.1", "0.3", "0.5", "0.7", "0.9"]
    
    # Calculate average success rates by temperature for each strategy
    success_rates = {
        "Parallel": [],
        "Serial": [],
        "Iterative": []
    }
    
    # Function to calculate average success rate from experiment data
    def extract_success_rate(data, temp):
        if "summary_by_temperature" in data and temp in data["summary_by_temperature"]:
            return data["summary_by_temperature"][temp].get("success_rate", 0)
        
        # Alternative extraction if different format
        problems = data.get("problems", {})
        if not problems:
            return 0
            
        successes = 0
        for problem_id, problem_data in problems.items():
            branches = problem_data.get("branches", [])
            for branch in branches:
                if branch.get("temperature") == float(temp) and branch.get("verified", False):
                    successes += 1
                    break
        
        return successes / len(problems) if problems else 0
    
    # Extract data for each strategy
    for temp in temperatures:
        success_rates["Parallel"].append(extract_success_rate(parallel_data, temp))
        success_rates["Serial"].append(extract_success_rate(serial_data, temp))
        success_rates["Iterative"].append(extract_success_rate(iterative_data, temp))
    
    # Set up the plot
    plt.figure(figsize=(12, 7))
    
    # Create groups for each temperature
    x = np.arange(len(temperatures))
    width = 0.25
    
    # Plot bars for each strategy
    plt.bar(x - width, success_rates["Parallel"], width, label='Parallel Sampling')
    plt.bar(x, success_rates["Serial"], width, label='Serial Sampling')
    plt.bar(x + width, success_rates["Iterative"], width, label='Iterative Refinement')
    
    # Add labels and title
    plt.xlabel('Temperature', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.title('Success Rate by Sampling Strategy and Temperature', fontsize=16, fontweight='bold')
    plt.xticks(x, temperatures)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add values above bars
    for i, v in enumerate(success_rates["Parallel"]):
        plt.text(i - width, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    
    for i, v in enumerate(success_rates["Serial"]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    
    for i, v in enumerate(success_rates["Iterative"]):
        plt.text(i + width, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Graph has been saved as '{output_file}'")


def analyze_solution_diversity(solutions: List[str], output_file: Optional[str] = "diversity_analysis.png"):
    """
    Analyze the diversity of solutions using TF-IDF and cosine similarity.
    
    Args:
        solutions: List of solution code strings
        output_file: Path to save the output figure
    """
    if len(solutions) < 2:
        print("Need at least 2 solutions to analyze diversity")
        return
    
    # Calculate TF-IDF vectors for each solution
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
    tfidf_matrix = vectorizer.fit_transform(solutions)
    
    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Create a heatmap of the similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim, annot=False, cmap='viridis_r', 
                xticklabels=range(1, len(solutions) + 1),
                yticklabels=range(1, len(solutions) + 1))
    
    plt.title('Solution Similarity Heatmap (Cosine Similarity)', fontsize=16, fontweight='bold')
    plt.xlabel('Solution Number', fontsize=14)
    plt.ylabel('Solution Number', fontsize=14)
    
    # Save the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Calculate diversity statistics
    avg_similarity = np.mean(cosine_sim[np.triu_indices(len(cosine_sim), k=1)])
    min_similarity = np.min(cosine_sim[np.triu_indices(len(cosine_sim), k=1)])
    max_similarity = np.max(cosine_sim[np.triu_indices(len(cosine_sim), k=1)])
    
    print(f"Graph has been saved as '{output_file}'")
    print(f"Diversity Statistics:")
    print(f"  Average Similarity: {avg_similarity:.4f}")
    print(f"  Minimum Similarity: {min_similarity:.4f}")
    print(f"  Maximum Similarity: {max_similarity:.4f}")
    print(f"  Diversity Score: {1 - avg_similarity:.4f} (higher is more diverse)")


def plot_unique_solutions_by_sample_count(data: Dict, difficulty: str, 
                                         output_file: Optional[str] = None):
    """
    Plot unique solutions vs sample count for different temperatures.
    
    Args:
        data: Experimental data
        difficulty: Difficulty level to plot (easy, medium, hard)
        output_file: Path to save the output figure
    """
    # Extract data for different sample counts and temperatures
    sample_counts = [1, 10, 50, 100]
    temperatures = ["0.1", "0.3", "0.5", "0.7", "0.9"]
    
    # Create a structure to hold unique solution counts
    unique_solutions = {temp: [] for temp in temperatures}
    
    # Process data to extract unique solution counts by sample count and temperature
    # This is a placeholder - actual implementation would depend on your data structure
    for sample_count in sample_counts:
        for temp in temperatures:
            # Calculate unique solutions for this sample count and temperature
            # Placeholder: replace with actual calculation
            unique_count = min(sample_count * float(temp) * 10, 100)  # Placeholder formula
            unique_solutions[temp].append(unique_count)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot line for each temperature
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i, temp in enumerate(temperatures):
        plt.plot(sample_counts, unique_solutions[temp], marker='o', linewidth=2, 
                 label=f"Temperature {temp}", color=colors[i % len(colors)])
    
    # Add labels and title
    plt.xlabel('Sample Count', fontsize=14)
    plt.ylabel('Unique Solutions', fontsize=14)
    plt.title(f'Unique Solutions vs. Sample Count ({difficulty.capitalize()} Problems)', 
              fontsize=16, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format the x-axis
    plt.xscale('log')
    plt.xticks(sample_counts, [str(x) for x in sample_counts])
    
    plt.tight_layout()
    
    # Save the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        output_file = f"unique_solutions_{difficulty}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Graph has been saved as '{output_file}'")


def main():
    """Example usage of data analysis functions"""
    # Example data for demonstration (would normally be loaded from files)
    example_data = {
        "summary_by_temperature": {
            "0.1": {
                "average_execution_time": 4.5,
                "average_solution_length": 1134,
                "success_rate": 0.85,
                "total_problems": 31
            },
            "0.3": {
                "average_execution_time": 4.0,
                "average_solution_length": 1144,
                "success_rate": 0.90,
                "total_problems": 31
            },
            "0.5": {
                "average_execution_time": 4.0,
                "average_solution_length": 1138,
                "success_rate": 0.92,
                "total_problems": 31
            },
            "0.7": {
                "average_execution_time": 5.0,
                "average_solution_length": 1103,
                "success_rate": 0.94,
                "total_problems": 31
            },
            "0.9": {
                "average_execution_time": 5.4,
                "average_solution_length": 1348,
                "success_rate": 0.95,
                "total_problems": 31
            }
        }
    }
    
    # Plot temperature metrics
    plot_temperature_metrics(example_data, "example_temperature_metrics.png")
    
    # Example solutions for diversity analysis
    example_solutions = [
        "def add(a, b):\n    return a + b",
        "def add(a, b):\n    result = a + b\n    return result",
        "def add(a, b):\n    # Add two numbers\n    return a + b",
        "def subtract(a, b):\n    return a - b",
        "def multiply(a, b):\n    return a * b"
    ]
    
    # Analyze solution diversity
    analyze_solution_diversity(example_solutions, "example_diversity_analysis.png")


if __name__ == "__main__":
    main()