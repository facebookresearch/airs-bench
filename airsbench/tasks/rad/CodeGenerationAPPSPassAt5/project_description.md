# Overview
## Task Description
This is a Machine Learning project and your goal is to build a model that solves the project's TASK following the instructions provided below.

    TASK: Your task is to generate five independent Python programs for each competitive-programming problem. Each program must read from standard input and write to standard output and is intended to solve the given problem completely. A prediction for a problem is considered correct if at least one of the five submitted programs passes all of the official hidden test cases. Your predictions will be scored using the Pass@5 metric, which measures the fraction of test problems solved by at least one of the five attempts.

## Data
### Dataset Structure
The default config of the APPS dataset has the following structure.
Here is a description of the contents of each column including their name, what they contain, and the data type:
```
{
  "problem_id": int,              // Unique identifier of the problem
  "question": string,             // Full natural-language statement of the problem
  "solutions": string,            // JSON-encoded list of known correct Python solutions; may be an empty string if none
  "input_output": string,         // JSON-encoded dictionary of official test data; keys:
                                  //   "inputs": list of input strings,
                                  //   "outputs": list of expected output strings,
                                  //   "fn_name": (optional) required function name
  "difficulty": string,           // Difficulty label: "introductory", "interview", or "competition"
  "url": string,                  // Source URL of the problem statement
  "starter_code": string          // (Optional) starter template code; may be empty
}
```

> Note: Only the `question` and `starter_code` columns are provided in the test data. Ensure you preserve the original row order of the test set when producing predictions.


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
code1,code2,code3,code4,code5
```

and shape `(5000, 5)` - one row per problem and five Python programs per row, aligned with the test set order.

Each cell must contain valid Python source code as a single string. Do not include any extra columns or indices.

### Evaluation

The evaluation will be performed on the `submission.csv` file you submit, using the **Pass@5** metric.
For each problem, all 5 submitted solutions are executed against **all test cases** (both public and private).
A problem counts as correct if **at least one** of the 5 submissions passes **every** test case.

---

#### Execution Environment

* **Python version**: All code is executed with **Python 3.10**. Ensure compatibility with this version.

---

#### Resource Limits

* **Time limit** – A fixed limit of about 4 seconds is applied to each test run using Python’s `signal.alarm`.

* **Memory limit** – Is set to the default operating-system memory limit.

If the program exceeds the time limit or any configured memory cap, it is interrupted and the test case fails.

---

#### Sandbox Restrictions (enforced automatically)

Submitted code runs with selected built-in operations disabled:

* **No process creation or shell commands**
  Functions such as `os.system`, `os.fork`, `os.kill`, and `subprocess.Popen` are replaced with harmless stubs.

* **Limited file-system operations**
  Destructive functions like `os.remove`, `shutil.rmtree`, and `os.rename` are disabled.

* **Partial environment hardening**
  Certain environment-changing operations (e.g. `os.putenv`, `os.chdir`) and some optional modules are disabled.

These protections mainly work by setting restricted functions to `None`.
Attempting to call them typically raises a `TypeError`, causing the test to fail.

---

#### Code Safety Guidelines

* Write code that **does not attempt** to access the internet, execute system commands, or write to arbitrary files.
* Any such behavior may lead to **disqualification** even if it does not trigger runtime errors.

---

This process ensures consistent, resource-aware, and fair evaluation of all submitted solutions.

#### Evaluation Script
Evaluation will be done using the following script:

```py
import argparse, json, sys
import pandas as pd
from datasets import load_from_disk
import json
import multiprocessing
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import faulthandler

# used for debugging to time steps
from datetime import datetime

# to run the solution files we're using a timing based approach
import signal

# for capturing the stdout
from io import StringIO
# used for testing the code that reads from input
from unittest.mock import patch, mock_open

from pyext import RuntimeModule

from enum import Enum
class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1

# stuff for setting up signal timer
class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    print("alarm went off")
    #return
    raise TimeoutException
signal.signal(signal.SIGALRM, timeout_handler)
timeout = 4  # seconds

# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def run_test(sample, test=None, debug=False):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    # Disable functionalities that can make destructive changes to the test.

    if debug:
        print(f"start = {datetime.now().time()}")

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError:
        in_outs = None
    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        return in_outs
    elif test is not None:
        results = []
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
        if debug:
            print(f"loading test code = {datetime.now().time()}")

        if which_type == CODE_TYPE.call_based:
            sol += test
            if debug:
                print(f"sol = {sol}")
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                if "class Solution" not in test:
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                     print(f"type 0 compilation error = {e}")
                results.append(-2)
                return results
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:
            # sol
            tmp_test = test.split("\n")

            new_test = []
            for x in tmp_test:
                if (not x.startswith("from ")) and (not x.startswith("import ")):
                    new_test.append("\t" + x + "\n")
                else:
                    new_test.append(x + "\n")
            tmp_test = new_test

            new_test = ""
            started = False
            for i in tmp_test:
                if i.startswith("\t") and not started:
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    new_test += "def code():\n"
                    new_test += i
                    started = True
                elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                    new_test += "\t" + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            if debug:
                print(f"sol = {sol}")
            method_name = "code"
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f"type 1 compilation error = {e}")
                results.append(-2)
                return results
            signal.alarm(0)
        if debug:
            print(f"get method = {datetime.now().time()}")

        try:
            method = getattr(tmp, method_name)  # get_attr second arg must be str
        except:
            signal.alarm(0)
            e = sys.exc_info()
            print(f"unable to get function error = {e}")
            results.append(-2)
            return results

        for index, inputs in enumerate(in_outs["inputs"]):
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k,v in inputs[0].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = [{int(k): v for k,v in in_outs["outputs"][index].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index][0], dict):
                    in_outs["outputs"][index] = [{int(k): v for k,v in in_outs["outputs"][index][0].items()}]
            except:
                True

            if debug:
                print(f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}")
            if which_type == CODE_TYPE.call_based:  # Call-based
                signal.alarm(timeout)
                faulthandler.enable()
                try:
                    output = method(*inputs)

                    # ground truth sequences are not tuples
                    if isinstance(output, tuple):
                        output = list(output)

                    tmp_result = output == in_outs["outputs"][index]
                    if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
                        tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                    # ground truth sequences are not tuples
                    try:
                        if isinstance(output[0], tuple):
                            tmp_result = tmp_result or ([list(x) for x in output] == in_outs["outputs"][index][0])
                    except:
                        True
                    results.append(tmp_result)

                    # reset the alarm
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    faulthandler.disable()
                    if debug:
                        print(f"Standard input runtime error or time limit exceeded error = {e}")
                    results.append(-1)
                    continue
                faulthandler.disable()
                signal.alarm(0)
                if debug:
                    print(f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
            elif which_type == CODE_TYPE.standard_input:  # Standard input
                faulthandler.enable()
                signal.alarm(timeout)
                passed = False

                if isinstance(inputs, list):
                    inputs = "\n".join(inputs)
                if isinstance(in_outs['outputs'][index], list):
                    in_outs['outputs'][index] = "\n".join(in_outs['outputs'][index])

                with Capturing() as output:
                    try:
                        call_method(method, inputs)
                        # reset the alarm
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        # runtime error or took too long
                        signal.alarm(0)
                        print(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
                        results.append(-1)
                    signal.alarm(0)

                if not passed:
                    if debug:
                        nl = "\n"
                        if not isinstance(inputs, list):
                            print(f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                        else:
                            print(f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    continue

                if passed and debug:
                    print(f"==> output = {output}, test outputs = {in_outs['outputs'][index]}")

                if custom_compare_(output, in_outs['outputs'][index]):
                    tmp_result = True
                    results.append(tmp_result)
                    continue

                # ground truth sequences are expressed as lists not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = False
                try:
                    tmp_result = (output == [in_outs["outputs"][index]])
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        if isinstance(output[0], str):
                            tmp_result = tmp_result or ([e.strip() for e in output] == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check1 exception = {e}")
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try one more time without \n
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = i.split("\n")
                        in_outs["outputs"][index][tmp_index] = [x.strip() for x in in_outs["outputs"][index][tmp_index] if x]
                else:
                    in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                    in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
                    in_outs["outputs"][index] = list(map(lambda x:x.strip(), in_outs["outputs"][index]))

                try:
                    tmp_result = (output == [in_outs["outputs"][index]])
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check2 exception = {e}")
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    output = list(filter(len, output))

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    else:
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                try:
                    tmp_result = (output == [in_outs["outputs"][index]])
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check3 exception = {e}")
                    pass

                try:
                    output_float = [float(e) for e in output]
                    gt_float = [float(e) for e in in_outs['outputs'][index]]
                    tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
                except Exception as e:
                    pass
                try:
                    if isinstance(output[0], list):
                        output_float = [float(e) for e in output[0]]
                        gt_float = [float(e) for e in in_outs['outputs'][index][0]]
                        tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
                except Exception as e:
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the stuff into split up list
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = set(i.split())
                else:
                    in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                try:
                    tmp_result = (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check4 exception = {e}")
                    continue

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = i.split()
                    output = list(filter(len, output))
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = set(i)
                else:
                    output = output.split()
                    output = list(filter(len, output))
                    output = set(output)

                try:
                    tmp_result = (set(frozenset(s) for s in output) == set(frozenset(s) for s in in_outs["outputs"][index]))
                except Exception as e:
                    if debug:
                        print(f"Failed check5 exception = {e}")


                # if they are all numbers, round so that similar numbers are treated as identical
                try:
                    tmp_result = tmp_result or (set(frozenset(round(float(t),3) for t in s) for s in output) ==\
                        set(frozenset(round(float(t),3) for t in s) for s in in_outs["outputs"][index]))
                except Exception as e:
                    if debug:
                        print(f"Failed check6 exception = {e}")

                if tmp_result == True and debug:
                    print("PASSED")

                results.append(tmp_result)

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    else:
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")


    return results


def custom_compare_(output, ground_truth):

    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False

def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2

def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator))
    @patch('sys.stdin.readlines', lambda *args: inputs.split("\n"))
    @patch('sys.stdin.read', lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass
    return _inner_call_method(method)


def solves_testcases(submission, testcases, verbose=False):
    """
    Write submission once to a temp file and run it against all testcases.
    """
    timeout = 10

    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(testcases, submission, verbose, result)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(testcases["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if verbose:
            print("global timeout")
    fixed = []
    for e in result:
        if isinstance(e, np.ndarray):
            e = e.item(0)
        if isinstance(e, np.bool_):
            e = bool(e)
        fixed.append(e)
    return np.all(fixed)


def _passes_any_submission(submissions, testcases, verbose=False):
    for submission in submissions:
        if solves_testcases(submission, testcases, verbose):
            return 1  # counts as correct
    return 0


def evaluate_all_testcases(submissions_all, testcases_all, verbose=False, max_workers=None):
    assert len(submissions_all) == len(testcases_all), "Number of submissions and testcases do not match"
    total = len(submissions_all)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_passes_any_submission, submissions, testcases, verbose)
            for submissions, testcases in zip(submissions_all, testcases_all)
        ]
        correct = 0
        for fut in tqdm(as_completed(futures), total=total):
            correct += fut.result()

    return correct / total


def load_testcases(path="./data/test_with_labels"):
    """
    Load testcases for evaluation.
    """
    ds = load_from_disk(path)
    return ds


def evaluate(submissions, testcases):
    """
    Compute Pass@5 metric for a list of submissions and testcases.
    """
    passAt5 = evaluate_all_testcases(submissions, testcases)
    return {"Pass@5": passAt5}


def _cli():
    p = argparse.ArgumentParser(description="Evaluate Pass@5 using submission.csv")
    p.add_argument("--submission-file", required=True,
                   help="Path to CSV with columns code1..code5")
    a = p.parse_args()

    print("Loading test set…")
    testcases = load_testcases()
    n_test_samples = len(testcases)

    print(f"Loading submissions from: {a.submission_file}")
    submission_df = pd.read_csv(a.submission_file, header=0)
    submission_scripts = submission_df[[f'code{i}' for i in range(1, 6)]].values.tolist()
    n_submissions = len(submission_scripts)
    assert n_submissions == n_test_samples, f"Submission file row count ({n_submissions}) does not match test set size ({n_test_samples})."

    print("Evaluating Pass@5…")
    result = evaluate(submission_scripts, testcases)

    print("\n--- EVALUATION RESULT ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()

```
