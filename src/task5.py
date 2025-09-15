import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# ClaMP Dataset Solution (Target: AUC >= 0.90)

def document_hyperparameter_tuning_clamp(train_df_path, test_df_path):
 
    hyperparameters = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'scale_pos_weight': 3.0  
    }
    return hyperparameters


def train_model_return_scores_clamp(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=3.0  
    )

    
    model.fit(X_train, y_train)

    
    prob_label_1 = model.predict_proba(test_df)[:, 1]

    
    result_df = pd.DataFrame({
        'index': test_df.index,
        'prob_label_1': prob_label_1
    })

    return result_df



# UNSW-NB15 Dataset Solution (Target: AUC >= 0.76)


def document_hyperparameter_tuning_unsw(train_df_path, test_df_path):
   
    hyperparameters = {
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42
    }
    return hyperparameters


def train_model_return_scores_unsw(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
   
    
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    if 'label' in cat_cols:
        cat_cols.remove('label')  

    
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)
        label_encoders[col] = le
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']

    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    
    model.fit(X_train, y_train)

    
    prob_label_1 = model.predict_proba(test_df)[:, 1]

    
    result_df = pd.DataFrame({
        'index': test_df.index,
        'prob_label_1': prob_label_1
    })

    return result_df