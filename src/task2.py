import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA




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
                 one_hot_encode_cols: list[str],
                 min_max_scale_cols: list[str],
                 n_components: int,
                 feature_engineering_functions: dict):
        # Store inputs as instance variables
        self.one_hot_encode_cols = one_hot_encode_cols
        self.min_max_scale_cols = min_max_scale_cols
        self.n_components = n_components
        self.feature_engineering_functions = feature_engineering_functions

        # Initialize encoders/scalers/PCA (will be fitted on training data only)
        self.ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=self.n_components, random_state=0)

   
    def one_hot_encode_columns_train(self, train_features: pd.DataFrame) -> pd.DataFrame:
        # Split into columns to encode and other columns
        cols_to_encode = train_features[self.one_hot_encode_cols].copy()
        other_cols = train_features.drop(columns=self.one_hot_encode_cols).copy()

        # Fit OHE on training data
        ohe_encoded_array = self.ohe_encoder.fit_transform(cols_to_encode)

        # Get new column names: "colname_category"
        ohe_column_names = self.ohe_encoder.get_feature_names_out(self.one_hot_encode_cols)

        # Create DataFrame with proper column names and index
        ohe_df = pd.DataFrame(ohe_encoded_array, columns=ohe_column_names, index=train_features.index)

        # Concatenate back with non-encoded columns
        result = pd.concat([ohe_df, other_cols], axis=1)
        return result

   

    def one_hot_encode_columns_test(self, test_features: pd.DataFrame) -> pd.DataFrame:
        # Split into columns to encode and other columns
        cols_to_encode = test_features[self.one_hot_encode_cols].copy()
        other_cols = test_features.drop(columns=self.one_hot_encode_cols).copy()

        # Transform using fitted encoder (DO NOT FIT AGAIN!)
        ohe_encoded_array = self.ohe_encoder.transform(cols_to_encode)

        # Get same column names as training
        ohe_column_names = self.ohe_encoder.get_feature_names_out(self.one_hot_encode_cols)

        # Create DataFrame
        ohe_df = pd.DataFrame(ohe_encoded_array, columns=ohe_column_names, index=test_features.index)

        # Concatenate back
        result = pd.concat([ohe_df, other_cols], axis=1)
        return result


    def min_max_scaled_columns_train(self, train_features: pd.DataFrame) -> pd.DataFrame:
        # Split into columns to scale and others
        cols_to_scale = train_features[self.min_max_scale_cols].copy()
        other_cols = train_features.drop(columns=self.min_max_scale_cols).copy()

        # Fit scaler on training data
        scaled_array = self.scaler.fit_transform(cols_to_scale)

        # Create DataFrame with original column names
        scaled_df = pd.DataFrame(scaled_array, columns=self.min_max_scale_cols, index=train_features.index)

        # Concatenate back
        result = pd.concat([scaled_df, other_cols], axis=1)
        return result
    

    
    def min_max_scaled_columns_test(self, test_features: pd.DataFrame) -> pd.DataFrame:
        # Split into columns to scale and others
        cols_to_scale = test_features[self.min_max_scale_cols].copy()
        other_cols = test_features.drop(columns=self.min_max_scale_cols).copy()

        # Transform using fitted scaler (DO NOT FIT AGAIN!)
        scaled_array = self.scaler.transform(cols_to_scale)

        # Create DataFrame
        scaled_df = pd.DataFrame(scaled_array, columns=self.min_max_scale_cols, index=test_features.index)

        # Concatenate back
        result = pd.concat([scaled_df, other_cols], axis=1)
        return result
    




    def pca_train(self, train_features: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Drop columns with ANY NA values in the TRAINING set
        self.train_columns_ = train_features.columns  # Save original columns for reference
        train_clean = train_features.dropna(axis=1)

        # Step 2: Store which columns were kept (for use in pca_test)
        self.pca_columns_ = train_clean.columns.tolist()

        # Step 3: Initialize and fit PCA
        self.pca = PCA(n_components=self.n_components, random_state=0)
        pca_result = self.pca.fit_transform(train_clean)

        # Step 4: Create DataFrame with correct column names and index
        column_names = [f"component_{i+1}" for i in range(self.n_components)]
        pca_df = pd.DataFrame(pca_result, columns=column_names, index=train_features.index)

        return pca_df

    def pca_test(self, test_features: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Use ONLY the columns that were kept during TRAINING
        # This ensures feature alignment and avoids "unknown feature" errors
        test_clean = test_features[self.pca_columns_]

        # Step 2: Transform using the fitted PCA model
        pca_result = self.pca.transform(test_clean)

        # Step 3: Create DataFrame with correct column names and index
        column_names = [f"component_{i+1}" for i in range(self.n_components)]
        pca_df = pd.DataFrame(pca_result, columns=column_names, index=test_features.index)

        return pca_df
    def feature_engineering_train(self, train_features: pd.DataFrame) -> pd.DataFrame:
        result = train_features.copy()

        for new_col_name, func in self.feature_engineering_functions.items():
            new_series = func(result)
            result[new_col_name] = new_series

        return result

    # def feature_engineering_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
    #     # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
    #     feature_engineered_dataset = pd.DataFrame()
    #     return feature_engineered_dataset
    def feature_engineering_test(self, test_features: pd.DataFrame) -> pd.DataFrame:
        result = test_features.copy()

        for new_col_name, func in self.feature_engineering_functions.items():
            new_series = func(result)
            result[new_col_name] = new_series

        return result

    # def preprocess_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
    #     # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
    #     preprocessed_dataset = pd.DataFrame()
    #     return preprocessed_dataset

    def preprocess_train(self, train_features: pd.DataFrame) -> pd.DataFrame:
        # Apply steps in order: OHE → Scaling → Feature Engineering (NO PCA)
        train_features = self.one_hot_encode_columns_train(train_features)
        train_features = self.min_max_scaled_columns_train(train_features)
        train_features = self.feature_engineering_train(train_features)
        return train_features
    
    # def preprocess_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
    #     # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
    #     preprocessed_dataset = pd.DataFrame()
    #     return preprocessed_dataset
    def preprocess_test(self, test_features: pd.DataFrame) -> pd.DataFrame:
        # Apply SAME steps as train, but using fitted transformers (no fitting!)
        test_features = self.one_hot_encode_columns_test(test_features)
        test_features = self.min_max_scaled_columns_test(test_features)
        test_features = self.feature_engineering_test(test_features)
        return test_features