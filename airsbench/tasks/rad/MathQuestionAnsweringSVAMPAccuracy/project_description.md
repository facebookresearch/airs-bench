# Overview
## Task Description
This is a Machine Learning project and your goal is to build a model that solves the project's TASK following the instructions provided below.

TASK: Your task is solve math world prolems. Each example presents a short story followed by a specific question. Your task is to read the text and predict the correct numerical answer. Your predictions will be scored against the `Answer` column of the test set.

## Data
### Dataset Structure
The default config of SVAMP has the following structure.
Here is a description of the contents of each column including their name, what they contain, and the data type:
```
{
  "ID": string,               # Unique identifier for the problem
  "Body": string,             # The narrative or context text describing the scenario
  "Question": string,         # The actual math question asked based on the Body
  "Equation": string,         # The ground-truth equation used to compute the answer
  "Answer": string,           # The correct numerical solution to the problem (as text)
  "Type": string,             # The problem category/type. Is one of ["Subtraction", "Addition", "Common-Division", "Multiplication"].
  "question_concat": string   # Concatenation of Body and Question into one text field
}
```

An example entry in the train dataset contains the following:
- `ID`: "chal-777`
- `Body`: "There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups"
- `Question`: "How big is each group of bananas?"
- `Equation`: "( 290.0 / 2.0 )"
- `Answer`: "145"
- `Type`: "Common-Division"
- `question_concat`: "There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups How big is each group of bananas?"

> Note: The scoring columns (`Equation`, `Answer`, `Type`) have been removed from the test data. Ensure you preserve the original row order of the test set when producing predictions.

Train data can be loaded using:
```python
from datasets import load_from_disk
train_dataset = load_from_disk('./data/train')
```

Test data can be loaded using:
```python
from datasets import load_from_disk
test_dataset = load_from_disk('./data/test')
```

### Submission file
The submission file should contain the selected answers for the test set. The submission file should be a csv file named `submission.csv` with the following header:
```
Answer
```

And it should be of shape `(300,1)` — one prediction per row, aligned with the test set order.

Each row must contain the numerical answer to the question. Do not include any extra columns or indices.

### Evalution
The evaluation will be performed on the `submission.csv` file you have submitted using the accuracy metric.

Evaluation will be done using the following script:

```py
#!/usr/bin/env python3
import argparse, json
import numpy as np
import pandas as pd
from datasets import load_from_disk


def load_test_answers(path='./data/test_with_labels'):
    """
    Load answers for evaluation (string labels).
    """
    ds = load_from_disk(path)
    if "Answer" not in ds.features:
        raise RuntimeError(
            "Test data does not contain 'Answer'. "
            "Use a labeled copy like './data/test_with_labels'."
        )
    answers = list(ds["Answer"])
    return answers


def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """
    correct = np.fromiter(
        (int(p) == int(y) for p, y in zip(predictions, labels)), dtype=bool
    )
    return {"Accuracy": float(correct.mean())}


def _cli():
    p = argparse.ArgumentParser(
        description="Evaluate predictions for SVAMP test split using Accuracy."
    )
    p.add_argument("--submission-file", default="submission.csv",
                   help="Path to CSV file containing predictions.")
    a = p.parse_args()

    print("Loading test set labels...")
    labels = load_test_answers()
    n_test_samples = len(labels)
    print(f"Loaded {n_test_samples} labels.")

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