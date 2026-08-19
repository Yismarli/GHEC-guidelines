import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
import pandas as pd


def single_model_comparison(X_train, X_test, y_train, y_test):
    """
    Compare performance of individual classification models.
    """
    classifiers = [
        XGBClassifier(),
        CatBoostClassifier(verbose=False),
        LGBMClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        HistGradientBoostingClassifier(),
        ExtraTreesClassifier(),
        DecisionTreeClassifier(),
        BaggingClassifier(),
        KNeighborsClassifier(),
        svm.SVC(probability=True),
        MLPClassifier(max_iter=500, solver='adam', hidden_layer_sizes=128),
        GaussianNB(),
        LinearRegression()  # Note: Will be converted to binary classification
    ]

    # Store predictions from all models
    y_preds = np.zeros((len(X_test), len(classifiers)))
    f1_scores = {}

    print("================== Individual Model Performance Start ==================")

    for idx, clf in enumerate(classifiers):
        print(f"Iteration {idx + 1}: {type(clf).__name__}")

        # Train model
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)

        # Convert LinearRegression output to binary classification
        if isinstance(clf, LinearRegression):
            y_pred = (y_pred > 0.5).astype(int)

        y_preds[:, idx] = y_pred

        # Calculate evaluation metrics
        balance_acc = balanced_accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        f1_scores[idx] = f1

        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn)  # True Positive Rate
        tnr = tn / (fp + tn)  # True Negative Rate

        print(f"F1: {f1:.6f}, Balanced Accuracy: {balance_acc:.6f}, "
              f"Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, "
              f"Recall: {recall:.6f}, TPR: {tpr:.6f}, TNR: {tnr:.6f}")

    print("================== Individual Model Performance End ==================")

    return classifiers, y_preds, f1_scores


def remove_worst_performer(classifiers, y_val_preds, y_test_preds, y_val, y_test):
    """
    Remove the worst performing model based on sub-layer evaluation.
    """
    # Get sorted results from sub-layer evaluation
    sorted_results = evaluate_sub_layer(classifiers, y_val_preds, y_test_preds, y_val, y_test)

    # Find the worst performer
    worst_performer_name = sorted_results[-1][0]
    worst_index = next(i for i, clf in enumerate(classifiers)
                       if type(clf).__name__ == worst_performer_name)

    # Remove the worst performer
    classifiers.pop(worst_index)
    y_val_preds = np.delete(y_val_preds, worst_index, axis=1)
    y_test_preds = np.delete(y_test_preds, worst_index, axis=1)

    return classifiers, y_val_preds, y_test_preds


def evaluate_sub_layer(classifiers, y_val_preds, y_test_preds, y_val, y_test):
    """
    Evaluate models in the sub-layer and sort by performance.
    """
    results = []

    for clf in classifiers:
        clf_name = type(clf).__name__

        # Train on validation predictions
        clf.fit(y_val_preds, y_val)

        # Make predictions on test set
        y_pred = clf.predict(y_test_preds)

        # Convert LinearRegression output to binary classification
        if isinstance(clf, LinearRegression):
            y_pred = (y_pred > 0.5).astype(int)

        # Calculate metrics
        balance_acc = balanced_accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (fp + tn)

        results.append((clf_name, f1, balance_acc, accuracy, precision, recall, tpr, tnr))

    # Sort by F1 score in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    print("Model ranking:", [(name, score) for name, score, *_ in sorted_results])

    return sorted_results


def ensemble_averaging(X_train, X_test, y_train, y_test):
    """
    Implement ensemble averaging (soft voting) approach.
    """
    classifiers, y_preds, f1_scores = single_model_comparison(X_train, X_test, y_train, y_test)

    # Sort indices by F1 score
    sorted_indices = sorted(f1_scores, key=f1_scores.get)

    best_f1 = -float('inf')
    deletion_history = []

    print("================== Ensemble Averaging (Soft Voting) Start ==================")

    # Try removing worst models one by one
    for i in range(len(classifiers) - 1):
        # Keep remaining models
        remaining_indices = sorted_indices[i:]

        # Calculate mean prediction
        mean_pred = y_preds[:, remaining_indices].mean(axis=1)
        mean_pred = (mean_pred > 0.5).astype(int)

        current_f1 = f1_score(y_test, mean_pred)

        # Update best F1 and deletion history
        if current_f1 > best_f1:
            best_f1 = current_f1
            deletion_history = [
                f"Model {type(classifiers[sorted_indices[k]]).__name__} "
                f"(F1: {f1_scores[sorted_indices[k]]:.4f})"
                for k in range(i)
            ]

        print(f"Removed {i} models, Current F1: {current_f1:.6f}, "
              f"Deletion History: {deletion_history}")

    print("================== Ensemble Averaging End ==================")
    return best_f1


def ensemble_voting(X_train, X_test, y_train, y_test):
    """
    Implement majority voting ensemble approach.
    """
    classifiers, y_preds, f1_scores = single_model_comparison(X_train, X_test, y_train, y_test)

    # Sort indices by F1 score
    sorted_indices = sorted(f1_scores, key=f1_scores.get)

    best_f1 = -float('inf')
    deletion_history = []

    print("================== Majority Voting Ensemble Start ==================")

    # Try removing worst models one by one
    for i in range(len(classifiers) - 1):
        # Keep remaining models
        remaining_indices = sorted_indices[i:]

        # Get predictions from remaining models
        y_vote = y_preds[:, remaining_indices]

        # Majority vote
        num_ones = y_vote.sum(axis=1)
        majority_vote = (num_ones > y_vote.shape[1] // 2).astype(int)

        current_f1 = f1_score(y_test, majority_vote)

        # Update best F1 and deletion history
        if current_f1 > best_f1:
            best_f1 = current_f1
            deletion_history = [
                f"Model {type(classifiers[sorted_indices[k]]).__name__} "
                f"(F1: {f1_scores[sorted_indices[k]]:.4f})"
                for k in range(i)
            ]

        print(f"Removed {i} models, Current F1: {current_f1:.6f}, "
              f"Deletion History: {deletion_history}")

    print("================== Majority Voting End ==================")
    return best_f1


def blending_ensemble(X_train, X_test, y_train, y_test):
    """
    Implement blending ensemble approach with holdout validation.
    """
    print("================== Blending Ensemble Start ==================")

    # Split training data for blending
    X_base, X_holdout, y_base, y_holdout = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )

    classifiers = [
        XGBClassifier(),
        CatBoostClassifier(verbose=False),
        LGBMClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        HistGradientBoostingClassifier(),
        ExtraTreesClassifier(),
        DecisionTreeClassifier(),
        BaggingClassifier(),
        KNeighborsClassifier(),
        svm.SVC(probability=True),
        MLPClassifier(max_iter=500, solver='adam', hidden_layer_sizes=128),
        GaussianNB(),
        LinearRegression()
    ]

    # Collect predictions from all models
    y_holdout_preds = np.zeros((len(X_holdout), len(classifiers)))
    y_test_preds = np.zeros((len(X_test), len(classifiers)))

    # Base layer training
    for idx, clf in enumerate(classifiers):
        clf.fit(X_base, y_base)

        # Predict on holdout set
        y_holdout_pred = clf.predict(X_holdout)
        y_holdout_preds[:, idx] = y_holdout_pred

        # Predict on test set
        y_test_pred = clf.predict(X_test)
        y_test_preds[:, idx] = y_test_pred

    # Remove worst performers iteratively
    while len(classifiers) > 0:
        classifiers, y_holdout_preds, y_test_preds = remove_worst_performer(
            classifiers, y_holdout_preds, y_test_preds, y_holdout, y_test
        )

    print("================== Blending Ensemble End ==================")


def stacking_ensemble(X_train, X_test, y_train, y_test):
    """
    Implement stacking ensemble approach with cross-validation.
    """
    print("================== Stacking Ensemble Start ==================")

    classifiers = [
        XGBClassifier(),
        CatBoostClassifier(verbose=False),
        LGBMClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        HistGradientBoostingClassifier(),
        ExtraTreesClassifier(),
        DecisionTreeClassifier(),
        BaggingClassifier(),
        KNeighborsClassifier(),
        svm.SVC(probability=True),
        MLPClassifier(max_iter=500, solver='adam', hidden_layer_sizes=128),
        GaussianNB(),
        LinearRegression()
    ]

    # Setup cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Collect out-of-fold predictions
    y_oof_preds = np.zeros((len(X_train), len(classifiers)))
    y_test_preds = np.zeros((len(X_test), len(classifiers)))

    for idx, clf in enumerate(classifiers):
        # Get out-of-fold predictions
        oof_preds = cross_val_predict(clf, X_train, y_train, cv=kf)
        y_oof_preds[:, idx] = oof_preds

        # Train on full training set and predict on test set
        clf.fit(X_train, y_train)
        y_test_preds[:, idx] = clf.predict(X_test)

    # Remove worst performers iteratively
    while len(classifiers) > 0:
        classifiers, y_oof_preds, y_test_preds = remove_worst_performer(
            classifiers, y_oof_preds, y_test_preds, y_train, y_test
        )

    print("================== Stacking Ensemble End ==================")


def comparative_trial_classification(X_train, X_test, y_train, y_test):
    """
    Main function to run all comparative trials for classification.
    """
    # Individual model comparison
    single_model_comparison(X_train, X_test, y_train, y_test)

    # Ensemble methods
    ensemble_averaging(X_train, X_test, y_train, y_test)
    ensemble_voting(X_train, X_test, y_train, y_test)
    blending_ensemble(X_train, X_test, y_train, y_test)
    stacking_ensemble(X_train, X_test, y_train, y_test)
