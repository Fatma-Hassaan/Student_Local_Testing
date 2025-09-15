import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.feature_selection import RFE

class ModelMetrics:
    def __init__(self, model_name: str, train_metrics: dict, test_metrics: dict, feature_importance_df: pd.DataFrame):
        self.model_name = model_name
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.feature_importance_df = feature_importance_df
        self.feat_name_col = "Feature"
        self.imp_col = "Importance"

def calculate_naive_metrics(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_targets: pd.Series,
    test_targets: pd.Series,
    naive_assumption: int
) -> ModelMetrics:
    train_preds = [naive_assumption] * len(train_targets)
    test_preds = [naive_assumption] * len(test_targets)

    train_probs = [1.0 if naive_assumption == 1 else 0.0] * len(train_targets)
    test_probs = [1.0 if naive_assumption == 1 else 0.0] * len(test_targets)

    train_metrics = {
        "accuracy": round(accuracy_score(train_targets, train_preds), 4),
        "recall": round(recall_score(train_targets, train_preds, zero_division=0), 4),
        "precision": round(precision_score(train_targets, train_preds, zero_division=0), 4),
        "fscore": round(f1_score(train_targets, train_preds, zero_division=0), 4),
        "fpr": 0.0,
        "fnr": 0.0,
        "roc_auc": round(roc_auc_score(train_targets, train_probs), 4)
    }

    test_metrics = {
        "accuracy": round(accuracy_score(test_targets, test_preds), 4),
        "recall": round(recall_score(test_targets, test_preds, zero_division=0), 4),
        "precision": round(precision_score(test_targets, test_preds, zero_division=0), 4),
        "fscore": round(f1_score(test_targets, test_preds, zero_division=0), 4),
        "fpr": 0.0,
        "fnr": 0.0,
        "roc_auc": round(roc_auc_score(test_targets, test_probs), 4)
    }

    naive_metrics = ModelMetrics("Naive", train_metrics, test_metrics, None)
    return naive_metrics


def calculate_logistic_regression_metrics(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_targets: pd.Series,
    test_targets: pd.Series,
    logreg_kwargs: dict
) -> tuple[ModelMetrics, LogisticRegression]:
    model = LogisticRegression(**logreg_kwargs)
    model.fit(train_features, train_targets)

    train_preds = model.predict(train_features)
    test_preds = model.predict(test_features)
    train_probs = model.predict_proba(train_features)[:, 1]
    test_probs = model.predict_proba(test_features)[:, 1]

    tn_train, fp_train, fn_train, tp_train = confusion_matrix(train_targets, train_preds).ravel()
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(test_targets, test_preds).ravel()

    fpr_train = fp_train / (fp_train + tn_train) if (fp_train + tn_train) > 0 else 0.0
    fnr_train = fn_train / (fn_train + tp_train) if (fn_train + tp_train) > 0 else 0.0
    fpr_test = fp_test / (fp_test + tn_test) if (fp_test + tn_test) > 0 else 0.0
    fnr_test = fn_test / (fn_test + tp_test) if (fn_test + tp_test) > 0 else 0.0

    train_metrics = {
        "accuracy": round(accuracy_score(train_targets, train_preds), 4),
        "recall": round(recall_score(train_targets, train_preds), 4),
        "precision": round(precision_score(train_targets, train_preds), 4),
        "fscore": round(f1_score(train_targets, train_preds), 4),
        "fpr": round(fpr_train, 4),
        "fnr": round(fnr_train, 4),
        "roc_auc": round(roc_auc_score(train_targets, train_probs), 4)
    }

    test_metrics = {
        "accuracy": round(accuracy_score(test_targets, test_preds), 4),
        "recall": round(recall_score(test_targets, test_preds), 4),
        "precision": round(precision_score(test_targets, test_preds), 4),
        "fscore": round(f1_score(test_targets, test_preds), 4),
        "fpr": round(fpr_test, 4),
        "fnr": round(fnr_test, 4),
        "roc_auc": round(roc_auc_score(test_targets, test_probs), 4)
    }

    rfe = RFE(estimator=LogisticRegression(**logreg_kwargs), n_features_to_select=10)
    rfe.fit(train_features, train_targets)

    selected_features = train_features.columns[rfe.support_]

    X_train_selected = train_features[selected_features]
    X_test_selected = test_features[selected_features]

    model_rfe = LogisticRegression(**logreg_kwargs)
    model_rfe.fit(X_train_selected, train_targets)

    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": model_rfe.coef_[0]
    })

    importance_df['abs_importance'] = importance_df['Importance'].abs()
    importance_df = importance_df.sort_values(by='abs_importance', ascending=False).drop(columns=['abs_importance'])

    importance_df["Importance"] = importance_df["Importance"].round(4)

    importance_df = importance_df.reset_index(drop=True)

    log_reg_metrics = ModelMetrics("Logistic Regression", train_metrics, test_metrics, importance_df)
    return log_reg_metrics, model


def calculate_decision_tree_metrics(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_targets: pd.Series,
    test_targets: pd.Series,
    tree_kwargs: dict
) -> tuple[ModelMetrics, DecisionTreeClassifier]:
    model = DecisionTreeClassifier(**tree_kwargs)
    model.fit(train_features, train_targets)

    train_preds = model.predict(train_features)
    test_preds = model.predict(test_features)
    train_probs = model.predict_proba(train_features)[:, 1]
    test_probs = model.predict_proba(test_features)[:, 1]

    tn_train, fp_train, fn_train, tp_train = confusion_matrix(train_targets, train_preds).ravel()
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(test_targets, test_preds).ravel()

    fpr_train = fp_train / (fp_train + tn_train) if (fp_train + tn_train) > 0 else 0.0
    fnr_train = fn_train / (fn_train + tp_train) if (fn_train + tp_train) > 0 else 0.0
    fpr_test = fp_test / (fp_test + tn_test) if (fp_test + tn_test) > 0 else 0.0
    fnr_test = fn_test / (fn_test + tp_test) if (fn_test + tp_test) > 0 else 0.0

    train_metrics = {
        "accuracy": round(accuracy_score(train_targets, train_preds), 4),
        "recall": round(recall_score(train_targets, train_preds), 4),
        "precision": round(precision_score(train_targets, train_preds), 4),
        "fscore": round(f1_score(train_targets, train_preds), 4),
        "fpr": round(fpr_train, 4),
        "fnr": round(fnr_train, 4),
        "roc_auc": round(roc_auc_score(train_targets, train_probs), 4)
    }

    test_metrics = {
        "accuracy": round(accuracy_score(test_targets, test_preds), 4),
        "recall": round(recall_score(test_targets, test_preds), 4),
        "precision": round(precision_score(test_targets, test_preds), 4),
        "fscore": round(f1_score(test_targets, test_preds), 4),
        "fpr": round(fpr_test, 4),
        "fnr": round(fnr_test, 4),
        "roc_auc": round(roc_auc_score(test_targets, test_probs), 4)
    }

    importance_df = pd.DataFrame({
        "Feature": train_features.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)

    importance_df["Importance"] = importance_df["Importance"].round(4)

    importance_df = importance_df.reset_index(drop=True)

    tree_metrics = ModelMetrics("Decision Tree", train_metrics, test_metrics, importance_df)
    return tree_metrics, model


def calculate_random_forest_metrics(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_targets: pd.Series,
    test_targets: pd.Series,
    rf_kwargs: dict
) -> tuple[ModelMetrics, RandomForestClassifier]:
    model = RandomForestClassifier(**rf_kwargs)
    model.fit(train_features, train_targets)

    train_preds = model.predict(train_features)
    test_preds = model.predict(test_features)
    train_probs = model.predict_proba(train_features)[:, 1]
    test_probs = model.predict_proba(test_features)[:, 1]

    tn_train, fp_train, fn_train, tp_train = confusion_matrix(train_targets, train_preds).ravel()
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(test_targets, test_preds).ravel()

    fpr_train = fp_train / (fp_train + tn_train) if (fp_train + tn_train) > 0 else 0.0
    fnr_train = fn_train / (fn_train + tp_train) if (fn_train + tp_train) > 0 else 0.0
    fpr_test = fp_test / (fp_test + tn_test) if (fp_test + tn_test) > 0 else 0.0
    fnr_test = fn_test / (fn_test + tp_test) if (fn_test + tp_test) > 0 else 0.0

    train_metrics = {
        "accuracy": round(accuracy_score(train_targets, train_preds), 4),
        "recall": round(recall_score(train_targets, train_preds), 4),
        "precision": round(precision_score(train_targets, train_preds), 4),
        "fscore": round(f1_score(train_targets, train_preds), 4),
        "fpr": round(fpr_train, 4),
        "fnr": round(fnr_train, 4),
        "roc_auc": round(roc_auc_score(train_targets, train_probs), 4)
    }

    test_metrics = {
        "accuracy": round(accuracy_score(test_targets, test_preds), 4),
        "recall": round(recall_score(test_targets, test_preds), 4),
        "precision": round(precision_score(test_targets, test_preds), 4),
        "fscore": round(f1_score(test_targets, test_preds), 4),
        "fpr": round(fpr_test, 4),
        "fnr": round(fnr_test, 4),
        "roc_auc": round(roc_auc_score(test_targets, test_probs), 4)
    }

    importance_df = pd.DataFrame({
        "Feature": train_features.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)

    importance_df["Importance"] = importance_df["Importance"].round(4)

    importance_df = importance_df.reset_index(drop=True)

    rf_metrics = ModelMetrics("Random Forest", train_metrics, test_metrics, importance_df)
    return rf_metrics, model