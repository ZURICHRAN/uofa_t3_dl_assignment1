# Perceptron Experiments

A minimal Perceptron implementation with optional L1/L2 regularization for the `diabetes_scale.txt` dataset. The script is a small teaching example that trains on the data and prints accuracy for different hyperparameters.

## Repository structure
- `Perceptron.py`: Contains the base `Perceptron` class, `PerceptronWithRegularization`, dataset loading logic, and several hyperparameter sweeps executed at import time.
- `diabetes_scale.txt` (not included): Expected dataset file in LIBSVM-like format where the first column is the label and the remaining `index:value` pairs represent eight features.

## Requirements
- Python 3.8+
- NumPy
- scikit-learn

You can install dependencies with:

```bash
pip install -r requirements.txt  # if you create one
# or
pip install numpy scikit-learn
```

## How to run
1. Place `diabetes_scale.txt` in the repository root.
2. Run the script:

```bash
python Perceptron.py
```

The script will:
- Load and split the dataset (80/20 train-test).
- Train perceptron models for multiple learning rates and iteration counts.
- Evaluate L1 and L2 regularization strengths separately.
- Print accuracy for each experiment to stdout.

## Notes and next steps
- The code executes all experiments on import; wrap logic in `if __name__ == "__main__":` if you want to import the classes without running experiments.
- Extend metrics (e.g., confusion matrix, precision/recall/F1) for deeper evaluation.
- Consider adding command-line arguments via `argparse` to customize hyperparameters.
- Add unit tests for `fit`/`predict` behavior if you plan to iterate on the models.
