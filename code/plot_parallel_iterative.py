import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# Set up plotting
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Roboto', 'Lato', 'DejaVu Sans']

# Load the data
parallel_df = pd.read_csv('./code/parallel_iterative_results/4omini_apps_parallel.csv')
iterative_df = pd.read_csv('./code/parallel_iterative_results/4omini_apps_iterative.csv')

# Map problem IDs to difficulty levels
def get_difficulty(problem_id):
    if 4000 <= problem_id <= 4030:
        return "Introductory"  # Easy
    elif 0 <= problem_id <= 30:
        return "Interview"     # Medium
    elif 3000 <= problem_id <= 3030:
        return "Competition"   # Hard
    else:
        return "Unknown"

# Add difficulty column to both dataframes
parallel_df['difficulty'] = parallel_df['problem_id'].apply(get_difficulty)
iterative_df['difficulty'] = iterative_df['problem_id'].apply(get_difficulty)

# For parallel sampling, we need to process solutions_generated
def is_successful(row):
    if pd.isna(row['verified']) or not row['verified']:
        return False
    return True

parallel_df['success'] = parallel_df.apply(is_successful, axis=1)

# Focus on 10-sample trials for parallel sampling
parallel_df_10samples = parallel_df[parallel_df['num_samples'] == 10]

# Group by difficulty and temperature to calculate success rates 
parallel_success_rates = parallel_df_10samples.groupby(['difficulty', 'temperature'])['success'].mean().reset_index()
iterative_success_rates = iterative_df.groupby(['difficulty', 'temperature'])['success'].mean().reset_index()

# Get unique difficulty levels
difficulty_levels = ["Introductory", "Interview", "Competition"]
difficulty_labels = {"Introductory": "Easy", "Interview": "Medium", "Competition": "Hard"}

# Define colors for parallel and iterative
parallel_color = "#636EFA"  # Blue
iterative_color = "#FF7F0E"  # Orange

# Create plots for each difficulty level
for difficulty in difficulty_levels:
    plt.figure(figsize=(10, 7))
    
    # Filter data for this difficulty
    parallel_subset = parallel_success_rates[parallel_success_rates['difficulty'] == difficulty]
    iterative_subset = iterative_success_rates[iterative_success_rates['difficulty'] == difficulty]
    
    # Check if we have data for each temperature
    temperatures = sorted(set(list(parallel_subset['temperature'].unique()) + list(iterative_subset['temperature'].unique())))
    
    # Create a list of success rates for each approach
    parallel_rates = []
    iterative_rates = []
    
    for temp in temperatures:
        # Get parallel success rate
        parallel_temp_data = parallel_subset[parallel_subset['temperature'] == temp]
        if not parallel_temp_data.empty:
            parallel_rates.append(parallel_temp_data['success'].values[0])
        else:
            parallel_rates.append(0)
            
        # Get iterative success rate
        iterative_temp_data = iterative_subset[iterative_subset['temperature'] == temp]
        if not iterative_temp_data.empty:
            iterative_rates.append(iterative_temp_data['success'].values[0])
        else:
            iterative_rates.append(0)
    
    # Plot the data
    plt.plot(temperatures, parallel_rates, 
             marker='o', linestyle='-', linewidth=3, markersize=10, alpha=0.85,
             label="Parallel Sampling", color=parallel_color)
    
    plt.plot(temperatures, iterative_rates, 
             marker='s', linestyle='-', linewidth=3, markersize=10, alpha=0.85,
             label="Iterative Sampling", color=iterative_color)
    
    # Add labels and styling
    plt.xticks(temperatures, [f"{t}" for t in temperatures], fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlabel("Temperature", fontsize=16, fontweight='medium')
    plt.ylabel("Success Rate", fontsize=16, fontweight='medium')
    plt.title(f"Parallel vs Iterative Sampling ({difficulty_labels[difficulty]})", fontsize=18, fontweight='bold')
    
    plt.legend(fontsize=13, loc='best', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"./imgs/{difficulty_labels[difficulty].lower()}_parallel10_vs_iterative.png", dpi=300, bbox_inches='tight')
    plt.show()

# Create combined plot with all difficulties
plt.figure(figsize=(12, 8))

# For each difficulty, plot both parallel and iterative
for i, difficulty in enumerate(difficulty_levels):
    # Create line styles for each difficulty
    linestyle = ['-', '--', ':'][i]
    
    # Filter data for this difficulty
    parallel_subset = parallel_success_rates[parallel_success_rates['difficulty'] == difficulty]
    iterative_subset = iterative_success_rates[iterative_success_rates['difficulty'] == difficulty]
    
    # Check if we have data for each temperature
    temperatures = sorted(set(list(parallel_subset['temperature'].unique()) + list(iterative_subset['temperature'].unique())))
    
    # Create a list of success rates for each approach
    parallel_rates = []
    iterative_rates = []
    
    for temp in temperatures:
        # Get parallel success rate
        parallel_temp_data = parallel_subset[parallel_subset['temperature'] == temp]
        if not parallel_temp_data.empty:
            parallel_rates.append(parallel_temp_data['success'].values[0])
        else:
            parallel_rates.append(0)
            
        # Get iterative success rate
        iterative_temp_data = iterative_subset[iterative_subset['temperature'] == temp]
        if not iterative_temp_data.empty:
            iterative_rates.append(iterative_temp_data['success'].values[0])
        else:
            iterative_rates.append(0)
    
    # Plot the data with difficulty-specific styling
    difficulty_label = difficulty_labels[difficulty]
    plt.plot(temperatures, parallel_rates, 
             marker='o', linestyle=linestyle, linewidth=3, markersize=10, alpha=0.85,
             label=f"Parallel - {difficulty_label}", color=parallel_color)
    
    plt.plot(temperatures, iterative_rates, 
             marker='s', linestyle=linestyle, linewidth=3, markersize=10, alpha=0.85,
             label=f"Iterative - {difficulty_label}", color=iterative_color)

# Add labels and styling for combined plot
plt.xticks(temperatures, [f"{t}" for t in temperatures], fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel("Temperature", fontsize=16, fontweight='medium')
plt.ylabel("Success Rate", fontsize=16, fontweight='medium')
plt.title(f"Parallel vs Iterative Sampling (All Difficulties)", fontsize=18, fontweight='bold')

plt.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

# Save the combined figure
plt.tight_layout()
plt.savefig("./imgs/all_difficulties_parallel_vs_iterative.png", dpi=300, bbox_inches='tight')
plt.show()