# Overview
## Task Description
This is a Machine Learning project and your goal is to build a model that solves the project's TASK following the instructions provided below.

TASK: Your task is to resolve pronoun references in natural language sentences. You will be given a sentence containing an ambiguous pronoun and a possible referent. Your goal is to predict whether the pronoun refers to the referent based on commonsense reasoning. Your predictions will be scored against the `label` column of the test.

## Data
### Dataset Structure
The default config of the WSC dataset has the following structure.
Here is a description of the contents of each column including their name, what they contain, and the data type:
```
{
    "text": string, # The passage containing an ambiguous pronoun and candidate antecedent
    "span1_index": int, # The token index of the first candidate antecedent in the passage
    "span2_index": int, # The token index of the pronoun in the passage
    "span1_text": string, # The text span corresponding to the candidate antecedent
    "span2_text": string, # The text span corresponding to the pronoun
    "label": int # Binary label: 1 if the pronoun refers to the antecedent, 0 otherwise
}
```

An example entry in the train dataset contains the following:
- `text`: "Mark told Pete many lies about himself, which Pete included in his book. He should have been more skeptical."
- `span1_index`: 0
- `span2_index`: 13
- `span1_text`: "Mark"
- `span2_text`: "He"
- `label`: 0

> Note: The task is to determine whether the pronoun (`span2_text`) refers to the antecedent candidate (`span1_text`). A label of `1` indicates coreference, while `0` indicates no coreference. The `label` column is only available in the test set. Ensure you preserve the original row order of the test set when producing predictions.


Train data can be loaded using:
```python
from datasets import load_from_disk
train_dataset = load_from_disk('./data/train')
```

Validation data can be loaded using:
```python
from datasets import load_from_disk
test_dataset = load_from_disk('./data/validation')
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

And it should be of shape `(104,1)` — one prediction per row, aligned with the test set order.

Each row must contain `1` pronoun (`span2_text`) refers to the antecedent candidate (`span1_text`), and `0` otherwise. Do not include any extra columns or indices.

### Evalution
The evaluation will be performed on the `submission.csv` file you have submitted using the Accuracy metric.

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
    if "label" not in ds.features:
        raise RuntimeError(
            "Test data does not contain 'label'. "
            "Use a labeled copy like './data/test_with_labels'."
        )
    labels = list(ds["label"])
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
        description="Evaluate predictions for TabFact test split using Accuracy."
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