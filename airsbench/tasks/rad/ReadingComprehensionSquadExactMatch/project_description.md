
# Overview
## Task Description
This is a Machine Learning project and your goal is to complete the project's TASK following the instructions provided below.

    TASK: This is a NLP task to perform Reading Comprehension on rajpurkar/squad.
    You will be provided with a `title`, `context` paragraph and a `question` about the context, and will need to extract the answer from the context and return it as a string. Your predictions will be scored against the `answers` column of the test set. More specifically, your prediction will be considered correct if it exactly matches one of the possible answers in the `answers` in the `text` field.


## Data
### Dataset Structure
The default config of rajpurkar/squad has the following structure.
Here is a description of the contents of each column including their name, what they contain, and the data type:
 ``` {
        'title': Value('string'), # title of the article
        'context': Value('string'), # context paragraph containing the answer
        'question': Value('string'), # question about the context
        'answers': { # possible answers to the question
            'text': List(Value('string')), # list of possible answers (This is the scoring column to predict)
            'answer_start': List(Value('int32')) # list of character positions where each answer starts in the context
        }
    }
```
Train data can be loaded using:
```python
    from datasets import load_from_disk
    load_from_disk('./data/train') 
```
Test data can be loaded using:
```python
    from datasets import load_from_disk
    load_from_disk('./data/test').
```
Note that the scoring column has been removed from the test data.
### Submission file
The submission file should be a csv file named `submission.csv` with the following header:
``` answers ```
Answers should have a single answer for each question in the test set (i.e. one string). Do not follow the the `answers` column structure in the original dataset, just provide a single string answer for each question.

And it should be of shape (10570,).

### Evalution
The evaluation will be performed on the `submission.csv` file you have submitted using the ExactMatch metric. Here is the evaluation script that will be used:
```python
def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """
    # Calculate exact match
    exact_matches = 0
    for pred, label_list in zip(predictions, labels):
        if format_prediction(pred) in label_list:
            exact_matches += 1
            continue
    
    exact_match_score = exact_matches / len(labels)

    return {"ExactMatch": exact_match_score}
```

Your csv submission will be loaded like this:
```python
import pandas as pd
submission_df = pd.read_csv(a.submission_file, header=0)
preds = submission_df.values.squeeze()
```