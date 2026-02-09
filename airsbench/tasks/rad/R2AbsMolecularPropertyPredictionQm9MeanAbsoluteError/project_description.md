# Overview
## Task Description
This is a Machine Learning project and your goal is to complete the project's TASK following the instructions provided below.

TASK: Your task is to predict a molecular property of small molecules which is known as the **squared spatial extent (R_2_Abs)**. This is a fundamental geometric property that quantifies the spatial distribution of electron density around the molecular center of mass. R_2_Abs represents the expectation value of the squared distance of electrons from the center of mass, providing a measure of molecular size and compactness. Values are given in Bohr² (atomic units of area), with larger values indicating more spatially extended molecules. This property is important for understanding molecular volume, shape, and interactions. Your predictions will be scored against the `R_2_Abs` column of the test set.

## Data
### Dataset Structure
The default config of qm9 has the following structure. Here is a description of the contents of each column including their name, what they contain, and the data type:
 ```
 {
  "atomic_numbers": List[int64],  # List of atomic numbers for each atom in the molecule (e.g., 1 for H, 6 for C, 8 for O)
  "pos": List[List[float32]],  # 3D Cartesian coordinates (x, y, z) for each atom, shape: [num_atoms, 3]
  "A": float32,  # Rotational constant A (GHz)
  "B": float32,  # Rotational constant B (GHz)
  "C": float32,  # Rotational constant C (GHz)
  "natoms": int64,  # Number of atoms in the molecule
  "tags": List[int64],  # Per-atom categorical labels
  "composition": List[int64],  # Vector encoding the count of each element in the molecule

}```

 An example entry of the train dataset contains the following:
 ```
 {
    'atomic_numbers': [6, 1, 1, 1, 1],  # Atomic numbers (e.g., C=6, H=1)
    'pos': [
        [-0.01269999984651804,1.085800051689148,0.00800000037997961],
        [0.002199999988079071,-0.006000000052154064,0.0020000000949949026],[1.0117000341415405,1.4637999534606934,0.0003000000142492354],
        [-0.5407999753952026,1.4474999904632568,-0.8766000270843506],
        [-0.5238000154495239,1.4378999471664429,0.9064000248908997]
    ],  # 3D coordinates for each atom
    'A': 157.711807,
    'B': 157.709976,
    'C': 157.706985,
    'natoms': 5,     # Number of atoms (int64)
    'tags': [2,2,2,2,2],
    'id': '1_167',
    'composition': [0,4,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    'R_2_Abs': 1173.51
}
```

> Note: The scoring column (`R_2_Abs`) has been removed from the test data. Ensure you preserve the original row order of the test set when producing predictions.

Train data can be loaded using:
```python
from datasets import load_from_disk
train_dataset = load_from_disk('./data/train')
```
Validation data can be loaded using:
```python
from datasets import load_from_disk
train_dataset = load_from_disk('./data/val')
```
Test data can be loaded using:
```python
from datasets import load_from_disk
test_dataset = load_from_disk('./data/test')
```

### Submission file
The submission file should contain the predicted values for the test set. The submission file should be a csv file named `submission.csv` with the following header:
```
R_2_Abs
```

And it should be of shape `(10000,1)` — one prediction per row, aligned with the test set order.

### Evalution
The evaluation will be performed on the `submission.csv` file you have submitted using the MeanAbsoluteError metric. Here is the evaluation script that will be used:
```
#!/usr/bin/env python3
import argparse, json, numpy as np, pandas as pd
# Conditional torch import - might be added by generated functions if needed
# import torch
from datasets import load_dataset, load_from_disk


from datasets import load_dataset
import numpy as np

def load_test_set():
    dataset = load_from_disk('./data/test_with_labels')
    return np.array(dataset["R_2_Abs"])


import torch
import numpy as np
from torchmetrics import MeanAbsoluteError

def evaluate(predictions, labels):
    # Convert numpy arrays to tensors if needed
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Ensure inputs are torch tensors with float type
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)

    predictions = predictions.float()
    labels = labels.float()

    # Initialize and compute metric
    metric = MeanAbsoluteError(num_outputs=1)
    score = metric(predictions, labels)

    return {"MeanAbsoluteError": score.item()}


import numpy as np
import random
import string

def generate_dummy_predictions(n_samples):
    # Given nature is "unknown", we'll return scalar values as a reasonable default
    predictions = np.random.uniform(0, 1, size=(n_samples, 1))
    return predictions.squeeze()



def _cli():
    p = argparse.ArgumentParser(
        description="Evaluate predictions for qm9 test split using MeanAbsoluteError."
    )
    p.add_argument("--dummy-submission", action="store_true",
                   help="Evaluate with randomly generated dummy predictions.")
    p.add_argument("--submission-file", default="submission.csv",
                   help="Path to CSV file containing predictions.")
    a = p.parse_args()

    print("Loading test set labels...")
    labels = load_test_set()
    n_test_samples = len(labels)
    print(f"Loaded {n_test_samples} labels.")

    if a.dummy_submission:
        print(f"Generating {n_test_samples} dummy predictions...")
        preds = generate_dummy_predictions(n_test_samples)
    else:
        print(f"Loading predictions from: {a.submission_file}")
        try:
            # Assuming CSV has no header and contains only prediction values
            # Adjust if your submission format is different (e.g., has headers, specific columns)
            submission_df = pd.read_csv(a.submission_file, header=0)
            preds = submission_df.values.squeeze()
            if preds.shape[0] != n_test_samples:
                raise ValueError(
                    f"Submission file row count ({preds.shape[0]}) "
                    f"does not match test set size ({n_test_samples})."
                )
            # Further shape validation could be added here against spec['shape_per_item']
            # e.g., if preds.shape[1:] != spec['shape_per_item']
        except FileNotFoundError:
            p.error(f"Submission file not found: {a.submission_file}")
        except Exception as e:
            p.error(f"Error loading submission_file: {e}")

    print("Evaluating predictions...")
    result = evaluate(preds, labels)

    print("\n--- EVALUATION RESULT ---")
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    _cli()
```
