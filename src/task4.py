import numpy as np
import pandas as pd
from sklearn.metrics import *

class ModelMetrics:
    def __init__(self, model_type:str,train_metrics:dict,test_metrics:dict,feature_importance_df:pd.DataFrame):
        self.model_type = model_type
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.feat_imp_df = feature_importance_df
        self.feat_name_col = "Feature"
        self.imp_col = "Importance"
    
    def add_train_metric(self,metric_name:str,metric_val:float):
        self.train_metrics[metric_name] = metric_val

    def add_test_metric(self,metric_name:str,metric_val:float):
        self.test_metrics[metric_name] = metric_val

    def __str__(self): 
        output_str = f"MODEL TYPE: {self.model_type}\n"
        output_str += f"TRAINING METRICS:\n"
        for key in sorted(self.train_metrics.keys()):
            output_str += f"  - {key} : {self.train_metrics[key]:.4f}\n"
        output_str += f"TESTING METRICS:\n"
        for key in sorted(self.test_metrics.keys()):
            output_str += f"  - {key} : {self.test_metrics[key]:.4f}\n"
        if self.feat_imp_df is not None:
            output_str += f"FEATURE IMPORTANCES:\n"
            for i in self.feat_imp_df.index:
                output_str += f"  - {self.feat_imp_df[self.feat_name_col][i]} : {self.feat_imp_df[self.imp_col][i]:.4f}\n"
        return output_str

def calculate_naive_metrics(train_features:pd.DataFrame, test_features:pd.DataFrame, train_targets:pd.Series, test_targets:pd.Series, naive_assumption:int) -> ModelMetrics:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task4.html and implement the function as described
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0
        }
    naive_metrics = ModelMetrics("Naive",train_metrics,test_metrics,None)
    return naive_metrics

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
def calculate_logistic_regression_metrics(train_features:pd.DataFrame, test_features:pd.DataFrame, train_targets:pd.Series, test_targets:pd.Series, logreg_kwargs) -> tuple[ModelMetrics,LogisticRegression]:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task4.html and implement the function as described
    model = LogisticRegression()
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }

    log_reg_importance = pd.DataFrame()
    log_reg_metrics = ModelMetrics("Logistic Regression",train_metrics,test_metrics,log_reg_importance)

    return log_reg_metrics,model

from sklearn.tree import DecisionTreeClassifier

def calculate_decision_tree_metrics( train_features: pd.DataFrame, test_features: pd.DataFrame, train_targets: pd.Series, test_targets: pd.Series, dt_kwargs) -> tuple[ModelMetrics, DecisionTreeClassifier]:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task4.html and implement the function as described
    model = DecisionTreeClassifier()
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }

    dt_importance = pd.DataFrame()
    dt_metrics = ModelMetrics("Decision Tree", train_metrics, test_metrics, dt_importance)

    return dt_metrics,model
from sklearn.ensemble import RandomForestClassifier
def calculate_random_forest_metrics(train_features:pd.DataFrame, test_features:pd.DataFrame, train_targets:pd.Series, test_targets:pd.Series, rf_kwargs) -> tuple[ModelMetrics,RandomForestClassifier]:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task4.html and implement the function as described
    model = RandomForestClassifier()
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }

    rf_importance = pd.DataFrame()
    rf_metrics = ModelMetrics("Random Forest",train_metrics,test_metrics,rf_importance)

    return rf_metrics,model

