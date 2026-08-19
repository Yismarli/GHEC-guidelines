"""
Base layer training for ensemble classifiers (binary/multi-class).
Uses 5-fold cross-validation to generate out-of-fold predictions and performance metrics.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier


def train_base_ensemble(X_train, X_test, y_train, y_test):
    """
    Train base classifiers with 5-fold CV.
    Returns:
        - node_predictions: validation predictions per model (DataFrame)
        - node_test_predictions: test predictions per model (DataFrame)
        - performance_metrics: DataFrame with balanced_accuracy, TPR, TNR per model
        - node_count: number of base models
        - val_y: combined validation labels
        - val_X: combined validation features
        - classifiers: list of (name, model) tuples
    """
    print("================== Base Ensemble Training ==================")

    classifiers = [
        ('XGB', XGBClassifier()),
        ('CatBoost', CatBoostClassifier(verbose=False)),
        ('LightGBM', LGBMClassifier()),
        ('AdaBoost', AdaBoostClassifier()),
        ('GradientBoost', GradientBoostingClassifier()),
        ('RandomForest', RandomForestClassifier()),
        ('ExtraTrees', ExtraTreesClassifier()),
        ('DecisionTree', DecisionTreeClassifier()),
        ('HistGradient', HistGradientBoostingClassifier()),
        ('Bagging', BaggingClassifier()),
        ('KNeighbors', KNeighborsClassifier()),
        ('SVM', svm.SVC()),
        ('MLP', MLPClassifier(max_iter=500)),
        ('GaussianNB', GaussianNB()),
        ('LinearRegression', LinearRegression())   # output converted to binary
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Storage
    predictions = {name: [] for name, _ in classifiers}
    val_X_list, val_y_list = [], []
    tpr_scores, tnr_scores, balanced_scores = [], [], []

    print("Starting cross-validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        for model_idx, (name, clf) in enumerate(classifiers):
            clf.fit(X_fold_train, y_fold_train)
            y_val_pred = clf.predict(X_fold_val)

            # Convert LinearRegression output to binary
            if isinstance(clf, LinearRegression):
                y_val_pred = (y_val_pred > 0.5).astype(int)

            # Compute metrics
            bal_acc = balanced_accuracy_score(y_fold_val, y_val_pred)
            tn, fp, fn, tp = confusion_matrix(y_fold_val, y_val_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (fp + tn) if (fp + tn) > 0 else 0

            tpr_scores.append((model_idx, tpr))
            tnr_scores.append((model_idx, tnr))
            balanced_scores.append((model_idx, bal_acc))

            predictions[name].append(y_val_pred)

        val_X_list.append(X_fold_val)
        val_y_list.append(y_fold_val)

    # Generate test predictions
    print("Generating test predictions...")
    node_test_predictions = pd.DataFrame()
    for name, clf in classifiers:
        test_pred = clf.predict(X_test)
        test_pred_df = pd.DataFrame(test_pred, columns=[name])
        node_test_predictions = pd.concat([node_test_predictions, test_pred_df], axis=1)

    # Combine validation predictions
    node_predictions = pd.DataFrame()
    for name, preds in predictions.items():
        combined = np.concatenate(preds)
        node_predictions[name] = combined

    # Average metrics per model
    avg_tpr = defaultdict(list); avg_tnr = defaultdict(list); avg_bal = defaultdict(list)
    for idx, val in tpr_scores: avg_tpr[idx].append(val)
    for idx, val in tnr_scores: avg_tnr[idx].append(val)
    for idx, val in balanced_scores: avg_bal[idx].append(val)

    performance_metrics = pd.DataFrame(columns=['Node', 'BalancedAccuracy', 'TPR', 'TNR'])
    for idx in range(len(classifiers)):
        performance_metrics.loc[idx] = {
            'Node': idx,
            'BalancedAccuracy': np.mean(avg_bal[idx]) if idx in avg_bal else np.nan,
            'TPR': np.mean(avg_tpr[idx]) if idx in avg_tpr else np.nan,
            'TNR': np.mean(avg_tnr[idx]) if idx in avg_tnr else np.nan
        }

    print("Performance metrics:")
    print(performance_metrics)

    # Combine validation data
    val_X_combined = pd.concat(val_X_list, ignore_index=True)
    val_y_combined = pd.concat(val_y_list, ignore_index=True)
    val_y_combined.columns = ['val_y']

    node_count = len(classifiers)

    return (node_predictions, node_test_predictions, performance_metrics,
            node_count, val_y_combined, val_X_combined, classifiers)