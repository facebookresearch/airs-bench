# Overview
## Task Description
This is a Machine Learning project and your goal is to build a model that solves the project's TASK following the instructions provided below.

TASK: Your task is to answer questions given a large context. You will be provided a title of a story and context surrounding its plot, then will be asked a question for which you should retrieve the answer from the context. Your predictions will be scored against a list of candidate answers provided in the `answers` column of the test.

## Data
### Dataset Structure
The default config of the DuoRC dataset has the following structure.
Here is a description of the contents of each column including their name, what they contain, and the data type:
```
{
    "plot_id": string, # The ID of the story
    "plot": string,  # The plot of the story
    "title": string,  # The title of the story
    "question": string,  # A question regarding a detail that may or may not be provided by the story
    "answers": list[string],   # A list of candidate answers to the question
    "no_answer": bool,   # A boolean indicating whether the question has no answer in the context.
}
```

An example entry in the train dataset contains the following:
- `plot_id`: "/m/03vyhn"
- `plot`: "Set in the second half of the 22nd century, Mars has been 84% terraformed, allowing humans to walk on the surface without pressure suits. Martian society has become matriarchal, with women in most positions of authority. The story concerns police officer Melanie Ballard (Natasha Henstridge), second in command of a team alongside Sergeant Jericho (Jason Statham) sent to a remote mining outpost to transport prisoner Desolation Williams (Ice Cube). Arriving at the remote mining town, Ballard finds all of the people missing. She learns that they had discovered an underground doorway created by an ancient Martian civilization."
- `title`: "Ghosts of Mars"
- `question`: "Who is colonized by a high tech company?"
- `answers`: ["Humans on Mars", "Mars"]
- `no_answer`: false

> Note: The `answers` and `no_answer` columns are not available in the test set provided. Ensure you preserve the original row order of the test set when producing predictions.


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
has_answer, answer
```

And it should be of shape `(15857, 2)` — one prediction per row, aligned with the test set order.

The `has_answer` column should produce a `true` or `false` value indicating whether the question has an answer provided in the supporting context.
The `answer` column should contain the answer if the `has_answer` column is true. If `has_answer` is false, any text in the `answer` column is ignored.
Do not include any extra columns or indices.

### Evalution
The evaluation will be performed on the `submission.csv` file you have submitted using the Accuracy metric. For each question, if the submitted `has_answer` is `false`, the prediction is only considered correct if the `no_answer` label is `true`. If the submitted `has_answer` is `true`, the prediction is only considered correct if the `no_answer` label is `false` and the submitted `answer` is an exact match of one of the `answers` in the label.

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
    labels = list(ds["answer"])
    no_answers = list(ds["no_answers"])
    return labels, no_answers

def is_correct_answer(submission, candidate_answers):
    submission = submission.lower()
    for candidate_answer in candidate_answers:
        if candidate_answer.lower() == submission:
            return True
    return False

def evaluate(submission_answers, submission_has_answers, label_answers, label_no_answers):
    """
    Returns a dict of metric_name -> value
    """
    correct = 0
    total = 0
    for submission_answer, submission_has_answer, label_answer, label_no_answer in zip(
        submission_answers, submission_has_answers, label_answers, label_no_answers
    ):
        total += 1
        if not submission_has_answer:
            correct += label_no_answer
        else:
            correct += is_correct_answer(answer(submission_answer, label_answer))
    return {"Accuracy": correct/total}


def _cli():
    p = argparse.ArgumentParser(
        description="Evaluate predictions for DuoRC test split using Accuracy."
    )
    p.add_argument("--submission-file", default="submission.csv",
                   help="Path to CSV file containing predictions.")
    a = p.parse_args()

    print("Loading test set labels...")
    label_answers, label_no_answers = load_test_answers()
    n_test_samples = len(label_answers)
    print(f"Loaded {n_test_samples} labels.")

    print(f"Loading predictions from: {a.submission_file}")
    try:
        # Assuming CSV has no header and contains only prediction values
        # Adjust if your submission format is different (e.g., has headers, specific columns)
        submission_df = pd.read_csv(a.submission_file, header=0)
        submission_answers = list(submission_df["answer"])
        submission_has_answers = list(submission_df["has_answer"])
        if len(submission_answers) != n_test_samples:
            raise ValueError(
                f"Submission file row count ({len(submission_answers)}) "
                f"does not match test set size ({n_test_samples})."
            )
    except FileNotFoundError:
        p.error(f"Submission file not found: {a.submission_file}")
    except Exception as e:
        p.error(f"Error loading submission_file: {e}")

    print("Evaluating predictions...")
    result = evaluate(submission_answers, submission_has_answers, label_answers, label_no_answers)

    print("\n--- EVALUATION RESULT ---")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    _cli()
```
