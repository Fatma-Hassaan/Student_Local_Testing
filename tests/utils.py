import unittest
from platform import python_version
import sklearn
import scipy
import pandas as pd
import yellowbrick
import numpy as np
import matplotlib

def compare_submission_to_answer(submission,answer,name):
    print(f"\nRUNNING TEST FOR {name}\n")
    if type(answer) != type(submission):
        print(f"\nFor {name} you submitted a {type(submission)} but the autograder expects a {type(answer)}\n")
        correct = False
    else:
        correct = answer==submission
        if correct == False:
            print(f"\nFailed Test for {name}\nExpected {answer} got {submission}")
    return correct

def compare_submission_to_answer_series(submission_series,answer_series,name):
    print(f"\nRUNNING TEST FOR {name}\n")
    if type(submission_series) != type(answer_series):
        print(f"For {name} you submitted a {type(submission_series)} but the autograder expects a {type(answer_series)}\n")
        sum=0
    else:
        if not submission_series.index.equals(answer_series.index):
            print(f"Your Index doesnt match the Answer Series in {name}\n")
        sum = 0
        if not submission_series.equals(answer_series):
            print(f"Your Series Does not Match the Answer Series in {name}\n")
        else: 
            sum = 1
    return sum  

def compare_submission_to_answer_df(submission_df,answer_df,name):
    print(f"\nRUNNING TEST FOR {name}\n")
    if type(submission_df) != type(answer_df):
        print(f"For {name} you submitted a {type(submission_df)} but the autograder expects a {type(answer_df)}\n")
        sum=0
    else:
        if len(submission_df.columns) != len(answer_df.columns):
            print(f"You do not have the correct number of columns in {name}\n")
        if not submission_df.index.equals(answer_df.index):
            print(f"Your Index doesnt match the Answer DF in {name}\n")
        if submission_df.shape[0] != answer_df.shape[0]:
            print(f"Your submission had {submission_df.shape[0]} rows and the answer expects {answer_df.shape[0]} rows in {name}\n")
        sum = 0
        if len(submission_df.columns) == len(answer_df.columns):
            counter = 0
            flag = 0
            for col in answer_df.columns:
              if (submission_df[col].equals(answer_df[col])) == False:
                if answer_df[col].dtype != submission_df[col].dtype:
                    try:
                        if (submission_df[col].astype(answer_df[col].dtype).equals(answer_df[col])) :
                            continue
                        else: 
                            print(f"{col}: submitted dtype {submission_df[col].dtype} expected dtype {answer_df[col].dtype}\n")
                    except:
                        print(f"{col}: submitted dtype {submission_df[col].dtype} expected dtype {answer_df[col].dtype}\n")
                else:
                    print(f"{col}\n")
                flag = 1
                counter += 1
            if flag == 0:
              sum=1
            else:
                print(f"Error with {counter} out of {len(answer_df.columns)} columns\n")
    return sum


# Expected versions from the ../env.yml file
EXPECTED_VERSIONS = {
    'python': '3.10.6',
    'sklearn': '1.4.0',
    'scipy': '1.12.0',
    'pandas': '2.2.0',
    'yellowbrick': '1.5',
    'numpy': '1.26.3',
    'matplotlib': '3.8.2'
}

class TestPackageVersions(unittest.TestCase):
    def test_python_version(self):
        self.assertEqual(EXPECTED_VERSIONS['python'], python_version())

    def test_sklearn_version(self):
        self.assertEqual(EXPECTED_VERSIONS['sklearn'], sklearn.__version__)

    def test_scipy_version(self):
        self.assertEqual(EXPECTED_VERSIONS['scipy'], scipy.__version__)

    def test_pandas_version(self):
        self.assertEqual(EXPECTED_VERSIONS['pandas'], pd.__version__)

    def test_yellowbrick_version(self):
        self.assertEqual(EXPECTED_VERSIONS['yellowbrick'], yellowbrick.__version__)

    def test_numpy_version(self):
        self.assertEqual(EXPECTED_VERSIONS['numpy'], np.__version__)

    def test_matplotlib_version(self):
        self.assertEqual(EXPECTED_VERSIONS['matplotlib'], matplotlib.__version__)

def run_version_checks():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPackageVersions)
    unittest.TextTestRunner().run(suite)
