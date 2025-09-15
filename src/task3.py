import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

class KmeansClustering:
    def __init__(self, random_state: int):
        """
        Initialize the KmeansClustering class.
        Store the random_state for reproducibility.
        """
        self.random_state = random_state
        self.kmeans_model = None  # Will store the fitted KMeans model
        self.optimal_k = None     # Will store the optimal number of clusters

    def kmeans_train(self, train_features: pd.DataFrame) -> list:
        """
        Train KMeans clustering on the training data.
        Use Yellowbrick's KElbowVisualizer to find optimal k in range [1, 10).
        Fit the final KMeans model with the optimal k.
        Return cluster IDs for each row in the training set.
        """
        # Initialize KMeans with random_state and n_init=10
        kmeans = KMeans(random_state=self.random_state, n_init=10)

        # Initialize Yellowbrick KElbowVisualizer to find optimal k in [1, 10)
        visualizer = KElbowVisualizer(kmeans, k=(1, 10))
        visualizer.fit(train_features)  # Fit to find optimal k

        # Get the optimal k value
        self.optimal_k = visualizer.elbow_value_

        # If no clear elbow is found (rare), default to 3
        if self.optimal_k is None:
            self.optimal_k = 3

        # Train final KMeans model with optimal k
        self.kmeans_model = KMeans(n_clusters=self.optimal_k, random_state=self.random_state, n_init=10)
        self.kmeans_model.fit(train_features)

        # Predict cluster IDs for training data
        cluster_ids = self.kmeans_model.predict(train_features).tolist()

        return cluster_ids

    def kmeans_test(self, test_features: pd.DataFrame) -> list:
        """
        Use the trained KMeans model to predict cluster IDs for the test set.
        Assumes kmeans_train() has already been called.
        """
        if self.kmeans_model is None:
            raise ValueError("Model not trained. Call kmeans_train() first.")

        # Predict cluster IDs for test data
        cluster_ids = self.kmeans_model.predict(test_features).tolist()

        return cluster_ids

    def train_add_kmeans_cluster_id_feature(self, train_features: pd.DataFrame) -> pd.DataFrame:
        """
        Add a new column 'kmeans_cluster_id' to the training DataFrame.
        The values are the cluster IDs from kmeans_train().
        """
        # Get cluster IDs
        cluster_ids = self.kmeans_train(train_features)

        # Create a copy to avoid modifying original
        df_copy = train_features.copy()
        df_copy['kmeans_cluster_id'] = cluster_ids

        return df_copy

    def test_add_kmeans_cluster_id_feature(self, test_features: pd.DataFrame) -> pd.DataFrame:
        """
        Add a new column 'kmeans_cluster_id' to the test DataFrame.
        The values are the cluster IDs from kmeans_test().
        """
        # Get cluster IDs (model should be trained by now via train_add_... or kmeans_train)
        cluster_ids = self.kmeans_test(test_features)

        # Create a copy to avoid modifying original
        df_copy = test_features.copy()
        df_copy['kmeans_cluster_id'] = cluster_ids

        return df_copy