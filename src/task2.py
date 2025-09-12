import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection
from sklearn.model_selection import train_test_split

# def tts(  dataset: pd.DataFrame,
#                        label_col: str, 
#                        test_size: float,
#                        should_stratify: bool,
#                        random_state: int) -> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
#     # TO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
#     train_features = pd.DataFrame()
#     test_features = pd.DataFrame()
#     train_labels = pd.Series()
#     test_labels = pd.Series()
#     return train_features,test_features,train_labels,test_labels



def tts(
    dataset: pd.DataFrame,
    label_col: str,
    test_size: float,
    should_stratify: bool,
    random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Step 1: Separate features (X) and labels (y)
    features = dataset.drop(columns=[label_col])   # All non-label columns
    labels = dataset[label_col]                    # Just the label column → Series!

    # Step 2: Split into train/test using scikit-learn
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels if should_stratify else None,  # Preserve class balance if needed
        random_state=random_state
    )

    # Step 3: Return the four components in exact order
    return train_features, test_features, train_labels, test_labels



class PreprocessDataset:
    def __init__(self, 
                 one_hot_encode_cols:list[str],
                 min_max_scale_cols:list[str],
                 n_components:int,
                 feature_engineering_functions:dict
                 ):
        # TODO: Add any state variables you may need to make your functions work
        return

    def one_hot_encode_columns_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        one_hot_encoded_dataset = pd.DataFrame()

        return one_hot_encoded_dataset

    def one_hot_encode_columns_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        one_hot_encoded_dataset = pd.DataFrame()

        return one_hot_encoded_dataset

    def min_max_scaled_columns_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        min_max_scaled_dataset = pd.DataFrame()
        return min_max_scaled_dataset

    def min_max_scaled_columns_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        min_max_scaled_dataset = pd.DataFrame()
        return min_max_scaled_dataset
    
    def pca_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        pca_dataset = pd.DataFrame()
        return pca_dataset

    def pca_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        pca_dataset = pd.DataFrame()
        return pca_dataset

    def feature_engineering_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        feature_engineered_dataset = pd.DataFrame()
        return feature_engineered_dataset
    
    def feature_engineering_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        feature_engineered_dataset = pd.DataFrame()
        return feature_engineered_dataset

    def preprocess_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        preprocessed_dataset = pd.DataFrame()
        return preprocessed_dataset
    
    def preprocess_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        preprocessed_dataset = pd.DataFrame()
        return preprocessed_dataset
