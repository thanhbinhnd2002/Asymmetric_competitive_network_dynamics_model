# Project: Gene Regulatory Network Analysis

## 📌 Overview
This project analyzes gene regulatory networks using competition-based simulations. The system compares the influence of different genes and determines their dominance based on a predefined set of parameters.

## 📁 Directory Structure
```
.
├── data/                # Raw input data
├── data_test/           # Test data
├── final_results/       # Processed results
├── results/             # Output results (Beta-specific subdirectories)
│   ├── Beta_1/
│   ├── Beta_2/
│   ├── Beta_3/
├── results_test_1/      # Test run results
├── compare.py           # Script for comparing results
├── Normal_Total_Support.py  # Sequential version of the algorithm
├── Parallel_Processing.py  # Optimized parallel version
├── requirement.txt      # Dependencies for running the project
├── readme.txt           # Project documentation
```

## 🚀 Getting Started
### 1️⃣ Install Dependencies
Run the following command to install all required libraries:
```bash
pip install -r requirements.txt
```

### 2️⃣ Running the Simulation
To execute the parallelized version of the simulation:
```bash
python Parallel_Processing.py
```
For the sequential version:
```bash
python Normal_Total_Support.py
```

## ⚙️ Key Features
- **Supports large-scale gene competition analysis**
- **Parallelized computation for improved performance**
- **Customizable parameters**: Gamma, Lambda, Decay, Iterations, Alpha, Beta
- **Structured output files for easy result analysis**

## 📊 Output Format
Each simulation run generates result files stored under `results/` directory with the following format:
```
Gen A    Gen B    Winner    Strength
Gene1    Gene2    Gen A     45
Gene3    Gene4    Draw      10
```

## ❗ Notes
- Ensure that your input files are formatted correctly before running the simulation.
- If running on a GPU, make sure CUDA and CuPy are properly installed.

## 🛠️ Author & Contributions
This project was developed and optimized for large-scale computational biology applications.
Feel free to contribute by submitting pull requests or reporting issues.

---
🚀 **Happy Computing!** 🚀

