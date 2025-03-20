# Adaptive Code Generation

This repository contains the code implementation for "Adaptive Code Generation: Multi-Branch Temperature and Sampling Analysis". The project explores how different sampling strategies, temperatures, and model configurations affect code generation performance across problems of varying complexity.

## Project Structure

The codebase is organized into several modules:

- **Sampling Strategies**:
  - `parallel_sampling.py` - Implementation of parallel independent sampling
  - `serial_sampling.py` - Implementation of serial sampling where each generation builds upon previous ones
  - `iterative_sampling.py` - Implementation of iterative refinement where a solution is refined repeatedly
  - `sequential_refinement.py` - Implementation of sequential sampling with error feedback

- **Analysis Tools**:
  - `solution_collector.py` - Tool for collecting solutions at different temperatures for analysis
  - `solution_verifier.py` - Utility for verifying solution correctness
  - `data_analysis.py` - Tools for analyzing and visualizing experimental results

## Getting Started

### Prerequisites

- Python 3.8+
- LLM API access (OpenAI, Together.ai, etc.)
- Required Python packages (see requirements below)

### Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install openai litellm datasets numpy matplotlib pandas seaborn scikit-learn
```

3. Set up your API keys (add to the appropriate scripts or use environment variables)
4. Download the datasets (APPS and/or LiveCodeBench)

### Running Experiments

Each sampling strategy can be run independently:

```bash
# Run parallel sampling experiments
python parallel_sampling.py

# Run serial sampling experiments
python serial_sampling.py

# Run iterative refinement experiments
python iterative_sampling.py

# Run sequential refinement with feedback
python sequential_refinement.py
```

### Analyzing Results

After running experiments, use the data analysis tools to analyze and visualize results:

```bash
# Analyze temperature effects
python data_analysis.py --experiment-dir experiments/your_experiment_name
```

## Components

### Parallel Sampling (`parallel_sampling.py`)

Implements parallel sampling where multiple independent solutions are generated at various temperatures. Features:
- Multi-temperature and multi-sample configurations
- Parallel solution generation
- Test-based verification

### Serial Sampling (`serial_sampling.py`)

Implements serial sampling where each generation builds upon previous generations. Features:
- Solutions reference and extend previous generations
- Multi-temperature configurations
- Parallel branch exploration

### Iterative Sampling (`iterative_sampling.py`)

Implements iterative refinement where a single solution path is iteratively improved. Features:
- Refinement-based approach
- Temperature control during iterations
- Solution verification

### Sequential Refinement (`sequential_refinement.py`)

Implements sequential sampling with error feedback. Features:
- Feedback-based refinement
- Error message incorporation
- Test case tracking

### Solution Collection and Analysis

- `solution_collector.py`: Collects solutions for qualitative analysis
- `solution_verifier.py`: Verifies solution correctness
- `data_analysis.py`: Provides tools for data visualization and analysis

## Datasets

The experiments use two main datasets:
- APPS (Automated Programming Progress Standard)
- LiveCodeBench

## Citation

If you use this code or reference our work, please cite:

```
@article{leevo2025adaptive,
  title={Adaptive Code Generation: Multi-Branch Temperature and Sampling Analysis},
  author={Lee, Daniel and Vo, Nicholas},
  journal={},
  year={2025}
}
```

## Acknowledgments

We thank Aakanksha Chowdhery and Jon Saad-Falcon for advising this project and providing helpful discussions, comments, and feedback. We also thank Together.ai and OpenAI for providing API credits that funded this project.
