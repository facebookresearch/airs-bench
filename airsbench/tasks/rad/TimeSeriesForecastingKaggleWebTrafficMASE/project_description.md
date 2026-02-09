
# Overview
## Task Description
This is a Machine Learning project and your goal is to complete the project's TASK following the instructions provided below.

    TASK: This is a Time Series task to perform time series forecasting (59
 forecasts ahead) on Monash-University/monash_tsf. You will be provided with time series data (`target`), and will need to forecast the future value (59
 values). Your predictions will be scored against the `label_target` column of the test set.

## Data
### Dataset Structure
The kaggle_web_traffic config of Monash-University/monash_tsf has the following structure.
Here is a description of the contents of each column including their name, what they contain, and the data type:
 ``` 
    {
        'label_target':  List(Value('float32')), # true future values for the time series (This is the scoring column to predict). It includes both the historical and future values concatenated together.
        'target':  List(Value('float32')), # historical values for the time series (a string representation of a list of floats)
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

!!!IMPORTANT NOTE!!!
    Some of the time series in this dataset contain NaN values. You should take this into account when building your model and making predictions.
### Submission file
The submission file should be a csv file named `submission.csv` with the following header:
``` label_target ```
Where each row contains your predicted future values (a string representation of a list of 59 floats) concatenated to the historical values all dumped as a string (using json.dumps for example) for the corresponding row in the test set.
And it should be of shape (145063,).

### Evaluation
The evaluation will be performed on the `submission.csv` file you have submitted using the average MASE (Mean Absolute Scaled Error) metric. Here is the evaluation script that will be used:
```python

from sktime.performance_metrics.forecasting import mean_absolute_scaled_error

def safe_literal_eval_with_nan(s):
    import ast
    import math
    s_fixed = s.replace('NaN', 'None')
    lst = ast.literal_eval(s_fixed)
    return lst

def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """
    mases = []
    test_ds = load_from_disk('./data/test_with_labels')
    train_targets = test_ds["target"]

    for pred, label, train_target in zip(predictions, labels, train_targets):
        try:
            pred = np.array(safe_literal_eval_with_nan(pred))
        except Exception as e:
            raise ValueError(f"Error parsing prediction: {pred}, with error {e}") from e
        label = np.array(label)
        
        if pred.shape != label.shape:
            raise ValueError(
                "Invalid sample: "
                f"Prediction shape {pred.shape} does not match "
                f"label shape {label.shape}"
            )

        train_size = np.array(train_target).shape[0]
        # remove first train_size elements from pred and label
        pred = pred[train_size:]
        label = label[train_size:]

        #find any nans in label
        mask = ~np.isnan(label)
        pred = pred[mask]
        label = label[mask]

        mases.append(mean_absolute_scaled_error(label, pred, train_target))

    return {"MASE": np.mean(mases)}
```

Your csv submission will be loaded like this:
```python
import pandas as pd
submission_df = pd.read_csv(a.submission_file, header=0)
preds = submission_df.values.squeeze()
```
    