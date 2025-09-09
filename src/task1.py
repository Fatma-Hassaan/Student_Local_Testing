import numpy as np
import pandas as pd


def find_data_type(dataset: pd.DataFrame, column_name: str) -> np.dtype:
    return dataset[column_name].dtype


def set_index_col(dataset: pd.DataFrame, index: pd.Series) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset.index = index
    return dataset


def reset_index_col(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.reset_index(drop=True)


def set_col_type(dataset: pd.DataFrame, column_name: str, new_col_type: type) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset[column_name] = dataset[column_name].astype(new_col_type)
    return dataset


def make_DF_from_2d_array(array_2d: np.array, column_name_list: list[str], index: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(array_2d, columns=column_name_list)
    df.index = index
    return df


def sort_DF_by_column(dataset: pd.DataFrame, column_name: str, descending: bool) -> pd.DataFrame:
    return dataset.sort_values(by=column_name, ascending=not descending)


def drop_NA_cols(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.dropna(axis=1)


def drop_NA_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.dropna(axis=0)


def make_new_column(dataset: pd.DataFrame, new_column_name: str, new_column_value: list) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset[new_column_name] = new_column_value
    return dataset


def left_merge_DFs_by_column(left_dataset: pd.DataFrame, right_dataset: pd.DataFrame, join_col_name: str) -> pd.DataFrame:
    return left_dataset.merge(right_dataset, on=join_col_name, how='left')


class simpleClass():
    def __init__(self, length: int, width: int, height: int):
        self.length = length
        self.width = width
        self.height = height


def find_dataset_statistics(dataset: pd.DataFrame, label_col: str) -> tuple[int, int, int, int, int]:
    n_records = len(dataset)
    n_columns = len(dataset.columns)
    n_negative = int((dataset[label_col] == 0).sum())  # Convert to Python int
    n_positive = int((dataset[label_col] == 1).sum())  # Convert to Python int
    perc_positive = int((n_positive / n_records) * 100)  # This is already an int

    return n_records, n_columns, n_negative, n_positive, perc_positive