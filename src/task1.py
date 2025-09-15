import pandas as pd
import numpy as np

def find_data_type(dataset: pd.DataFrame, column_name: str) -> np.dtype:
    """
    Returns the data type of the specified column.
    """
    return dataset[column_name].dtype

def set_index_col(dataset: pd.DataFrame, index: pd.Series) -> pd.DataFrame:
    """
    Sets the index of the dataset to be the provided series.
    """
    # Create a copy to avoid modifying the original
    df_copy = dataset.copy()
    df_copy.index = index
    return df_copy

def reset_index_col(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Resets the index of the dataset from 0 to n-1, dropping the old index.
    """
    return dataset.reset_index(drop=True)

def set_col_type(dataset: pd.DataFrame, column_name: str, new_col_type: type) -> pd.DataFrame:
    """
    Changes the data type of the specified column.
    """
    # Create a copy to avoid modifying the original
    df_copy = dataset.copy()
    df_copy[column_name] = df_copy[column_name].astype(new_col_type)
    return df_copy

def make_DF_from_2d_array(array_2d: np.array, column_name_list: list[str], index: pd.Series) -> pd.DataFrame:
    """
    Creates a DataFrame from a 2D array, with specified column names and index.
    """
    return pd.DataFrame(data=array_2d, columns=column_name_list, index=index)

def sort_DF_by_column(dataset: pd.DataFrame, column_name: str, descending: bool) -> pd.DataFrame:
    """
    Sorts the DataFrame by the specified column, in ascending or descending order.
    """
    return dataset.sort_values(by=column_name, ascending=not descending)

def drop_NA_cols(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Drops any columns that contain NA values.
    """
    return dataset.dropna(axis=1)

def drop_NA_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Drops any rows that contain NA values.
    """
    return dataset.dropna(axis=0)

def make_new_column(dataset: pd.DataFrame, new_column_name: str, new_column_value: list) -> pd.DataFrame:
    """
    Adds a new column to the DataFrame using the provided list of values.
    """
    # Create a copy to avoid modifying the original
    df_copy = dataset.copy()
    df_copy[new_column_name] = new_column_value
    return df_copy

def left_merge_DFs_by_column(left_dataset: pd.DataFrame, right_dataset: pd.DataFrame, join_col_name: str) -> pd.DataFrame:
    """
    Performs a left join of two DataFrames on the specified column.
    """
    return left_dataset.merge(right_dataset, on=join_col_name, how='left')

class simpleClass():
    def __init__(self, length: int, width: int, height: int):
        """
        Initializes the class with instance variables.
        """
        self.length = length
        self.width = width
        self.height = height

def find_dataset_statistics(dataset: pd.DataFrame, label_col: str) -> tuple[int, int, int, int, int]:
    """
    Calculates summary statistics for the dataset based on the label column.
    """
    n_records = len(dataset)
    n_columns = len(dataset.columns)
    n_negative = int((dataset[label_col] == 0).sum())
    n_positive = int((dataset[label_col] == 1).sum())
    perc_positive = int((n_positive / n_records) * 100) # Truncate after decimal

    return n_records, n_columns, n_negative, n_positive, perc_positive