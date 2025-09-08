import numpy as np
import pandas as pd
import sklearn.cluster
import yellowbrick.cluster

class KmeansClustering:
    def __init__(self, 
                 random_state: int
                ):
        # TODO: Add any state variables you may need to make your functions work
        pass

    def kmeans_train(self,train_features:pd.DataFrame) -> list:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task3.html and implement the function as described
        cluster_ids = list()
        return cluster_ids

    def kmeans_test(self,test_features:pd.DataFrame) -> list:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task3.html and implement the function as described
        cluster_ids = list()
        return cluster_ids

    def train_add_kmeans_cluster_id_feature(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task3.html and implement the function as described
        output_df = pd.DataFrame()
        return output_df

    def test_add_kmeans_cluster_id_feature(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task3.html and implement the function as described
        output_df = pd.DataFrame()
        return output_df