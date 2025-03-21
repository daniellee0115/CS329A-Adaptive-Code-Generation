import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# Set up plotting
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Roboto', 'Lato', 'DejaVu Sans']

# Load the data
parallel_df = pd.read_csv('./code/parallel_serial_results/4omini_lcb_parallel.csv')
serial_df = pd.read_csv('./code/parallel_serial_results/4omini_lcb_serial.csv')

# Define difficulty level mapping
difficulty_mapping = {
    "Difficulty.EASY": "Easy",
    "Difficulty.MEDIUM": "Medium",
    "Difficulty.HARD": "Hard"
}

# Process parallel data
parallel_df['difficulty_str'] = parallel_df['Difficulty'].map(difficulty_mapping)
parallel_df['success'] = parallel_df.apply(lambda row: row['verified'] == True, axis=1)

# Process serial data
serial_df['difficulty_str'] = serial_df['difficulty'].map(difficulty_mapping)
serial_df['success'] = serial_df.apply(lambda row: row['verified'] == True, axis=1)

# Focus on 10-sample trials for parallel sampling
parallel_df_10samples = parallel_df[parallel_df['num_samples'] == 10]

# Filter for temperatures from 0.1 to 0.9
parallel_df_10samples = parallel_df_10samples[
    (parallel_df_10samples['temperature'] >= 0.1) & 
    (parallel_df_10samples['temperature'] <= 0.9)
]
serial_df = serial_df[
    (serial_df['temperature'] >= 0.1) & 
    (serial_df['temperature'] <= 0.9)
]

# Group by difficulty and temperature to calculate success rates 
parallel_success_rates = parallel_df_10samples.groupby(['difficulty_str', 'temperature'])['success'].mean().reset_index()
serial_success_rates = serial_df.groupby(['difficulty_str', 'temperature'])['success'].mean().reset_index()

# Get unique difficulty levels
difficulty_levels = ["Easy", "Medium", "Hard"]

# Define colors for parallel and serial
parallel_color = "#636EFA"  # Blue
serial_color = "#FF7F0E"    # Orange

# Create plots for each difficulty level
for difficulty in difficulty_levels:
    plt.figure(figsize=(10, 7))
    
    # Filter data for this difficulty
    parallel_subset = parallel_success_rates[parallel_success_rates['difficulty_str'] == difficulty]
    serial_subset = serial_success_rates[serial_success_rates['difficulty_str'] == difficulty]
    
    # Skip if no data for this difficulty
    if parallel_subset.empty and serial_subset.empty:
        print(f"No data for difficulty: {difficulty}")
        continue
    
    # Check if we have data for each temperature
    temperatures = sorted(set(list(parallel_subset['temperature'].unique()) + list(serial_subset['temperature'].unique())))
    
    # Create a list of success rates for each approach
    parallel_rates = []
    serial_rates = []
    
    for temp in temperatures:
        # Get parallel success rate
        parallel_temp_data = parallel_subset[parallel_subset['temperature'] == temp]
        if not parallel_temp_data.empty:
            parallel_rates.append(parallel_temp_data['success'].values[0])
        else:
            parallel_rates.append(0)
            
        # Get serial success rate
        serial_temp_data = serial_subset[serial_subset['temperature'] == temp]
        if not serial_temp_data.empty:
            serial_rates.append(serial_temp_data['success'].values[0])
        else:
            serial_rates.append(0)
    
    # Plot the data
    plt.plot(temperatures, parallel_rates, 
             marker='o', linestyle='-', linewidth=3, markersize=10, alpha=0.85,
             label="Parallel Sampling", color=parallel_color)
    
    plt.plot(temperatures, serial_rates, 
             marker='s', linestyle='-', linewidth=3, markersize=10, alpha=0.85,
             label="Serial Sampling", color=serial_color)
    
    # Add labels and styling
    plt.xticks(temperatures, [f"{t}" for t in temperatures], fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlabel("Temperature", fontsize=16, fontweight='medium')
    plt.ylabel("Success Rate", fontsize=16, fontweight='medium')
    plt.title(f"Parallel vs Serial Sampling ({difficulty})", fontsize=18, fontweight='bold')
    
    plt.legend(fontsize=13, loc='best', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"./imgs/lcb_{difficulty.lower()}_parallel_vs_serial.png", dpi=300, bbox_inches='tight')
    plt.show()

# Create combined plot with all difficulties
plt.figure(figsize=(12, 8))

# For each difficulty, plot both parallel and serial
for i, difficulty in enumerate(difficulty_levels):
    # Create line styles for each difficulty
    linestyle = ['-', '--', ':'][i]
    
    # Filter data for this difficulty
    parallel_subset = parallel_success_rates[parallel_success_rates['difficulty_str'] == difficulty]
    serial_subset = serial_success_rates[serial_success_rates['difficulty_str'] == difficulty]
    
    # Skip if no data for this difficulty
    if parallel_subset.empty and serial_subset.empty:
        continue
    
    # Check if we have data for each temperature
    temperatures = sorted(set(list(parallel_subset['temperature'].unique()) + list(serial_subset['temperature'].unique())))
    
    # Create a list of success rates for each approach
    parallel_rates = []
    serial_rates = []
    
    for temp in temperatures:
        # Get parallel success rate
        parallel_temp_data = parallel_subset[parallel_subset['temperature'] == temp]
        if not parallel_temp_data.empty:
            parallel_rates.append(parallel_temp_data['success'].values[0])
        else:
            parallel_rates.append(0)
            
        # Get serial success rate
        serial_temp_data = serial_subset[serial_subset['temperature'] == temp]
        if not serial_temp_data.empty:
            serial_rates.append(serial_temp_data['success'].values[0])
        else:
            serial_rates.append(0)
    
    # Plot the data with difficulty-specific styling
    plt.plot(temperatures, parallel_rates, 
             marker='o', linestyle=linestyle, linewidth=3, markersize=10, alpha=0.85,
             label=f"Parallel - {difficulty}", color=parallel_color)
    
    plt.plot(temperatures, serial_rates, 
             marker='s', linestyle=linestyle, linewidth=3, markersize=10, alpha=0.85,
             label=f"Serial - {difficulty}", color=serial_color)

# Add labels and styling for combined plot
plt.xticks(sorted(set(temperatures)), [f"{t}" for t in sorted(set(temperatures))], fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel("Temperature", fontsize=16, fontweight='medium')
plt.ylabel("Success Rate", fontsize=16, fontweight='medium')
plt.title(f"Parallel vs Serial Sampling (All Difficulties)", fontsize=18, fontweight='bold')

plt.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

# Save the combined figure
plt.tight_layout()
plt.savefig("./imgs/lcb_all_difficulties_parallel_vs_serial.png", dpi=300, bbox_inches='tight')
plt.show()