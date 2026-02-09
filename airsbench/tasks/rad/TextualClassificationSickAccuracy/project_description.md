
# Overview
## Task Description
This is a Machine Learning project and your goal is to complete the project's TASK following the instructions provided below.

    TASK: This is a NLP task to perform textual-entailment on RobZamp/sick. You will be provided with a pair of sentences, `sentence_A` and `sentence_B`, and will need to predict the entailment between the two (0 = entailment, 1 = neutral , 2 = contradiction). Your predictions will be scored against the `label` column of the test set.

## Data
### Dataset Structure
The default config of RobZamp/sick has the following structure.
Here is a description of the contents of each column including their name, what they contain, and the data type:
 ``` 
    {
        'label': 'label': ClassLabel(names=['entailment', 'neutral', 'contradiction']), # the entailment between the two sentences (0 = entailment, 1 = neutral , 2 = contradiction)
        'sentence_A': Value('string'), # first sentence of the sentence pair
        'sentence_B': Value('string') # second sentence of the sentence pair 
    } 
```
Train data can be loaded using:
```python
    from datasets import load_from_disk
    load_from_disk('./data/train') 
```
Validation data can be loaded using:
```python
    from datasets import load_from_disk
    load_from_disk('./data/validation')
```
Test data can be loaded using:
```python
    from datasets import load_from_disk
    load_from_disk('./data/test').
```
Note that the scoring column has been removed from the test data.
### Submission file
The submission file should be a csv file named `submission.csv` with the following header:
``` label ```
Where each row contains your predicted class for the corresponding row in the test set.
And it should be of shape (4906,).

### Evalution
The evaluation will be performed on the `submission.csv` file you have submitted using the Accuracy metric. Here is the evaluation script that will be used:
```python
from sklearn.metrics import accuracy_score

def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """
    acc = accuracy_score(predictions, labels)
    return {"Accuracy": acc}
```

Your csv submission will be loaded like this:
```python
import pandas as pd
submission_df = pd.read_csv(a.submission_file, header=0)
preds = submission_df.values.squeeze()
```
    