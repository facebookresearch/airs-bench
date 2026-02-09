# Overview
## Task Description
This is a Machine Learning project and your goal is to build a model that solves the project's TASK following the instructions provided below.

TASK: Your task is to resolve ambiguous references in natural language sentences. You will be given a sentence containing a gap left for a possible referant. Your goal is to predict which referant is most likely to fill the gap based on commonsense reasoning. Your predictions will be scored against the `answer` column of the test.

## Data
### Dataset Structure
The default config of the Winogrande dataset has the following structure.
Here is a description of the contents of each column including their name, what they contain, and the data type:
```
{
    "sentence": string, # A full sentence containing an ambiguous referant and two candidate antecedents. The ambiguous referent is represented by an underscore in the sentence.
    "option1": string,  # The first candidate antecedent mentioned in the sentence
    "option2": string,  # The second candidate antecedent mentioned in the sentence
    "answer": string,   # The index of the correct antecedent from the two options that the pronoun refers to (either "1" if the correct antecedent is option1 or "2" if the correct antecedent is option2)
}
```

An example entry in the train dataset contains the following:
- `sentence`: "Ian volunteered to eat Dennis's menudo after already having a bowl because _ despised eating intestine."
- `option1`: "Ian"
- `option2`: "Dennis"
- `answer`: "2"

> Note: The `answer` column is not available in the test set provided. Ensure you preserve the original row order of the test set when producing predictions.


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

And it should be of shape `(1531, 1)` — one prediction per row, aligned with the test set order.

Each row must contain the string "1" if the correct antecedent is `option1` or string "2" if the correct antecedent is `option2`. Do not include any extra columns or indices.

### Evalution
The evaluation will be performed on the `submission.csv` file you have submitted using the Accuracy metric.

Evaluation will be done using the following script:

```py
import argparse, json
import numpy as np
import pandas as pd
from datasets import load_from_disk


def load_test_answers(path='./data/test_with_labels'):
    """
    Load answers for evaluation (string labels).
    """
    ds = load_from_disk(path)
    if "answer" not in ds.features:
        raise RuntimeError(
            "Test data does not contain 'label'. "
            "Use a labeled copy like './data/test_with_labels'."
        )
    labels = list(ds["answer"])
    return labels


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
        description="Evaluate predictions for Winogrande test split using Accuracy."
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
