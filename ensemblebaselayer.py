import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)
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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


def ensemble_base_layer_training(X_train, X_test, y_train, y_test):
    """
    Train base layer models for ensemble with cross-validation.
    Returns predictions, feature importance, and other metadata.
    """
    print("================== Ensemble Base Layer Training Start ==================")

    # Define classifiers with names
    classifiers = [
        ('XGB', XGBClassifier()),
        ('CatBoost', CatBoostClassifier(verbose=False)),
        ('LightGBM', LGBMClassifier()),
        ('AdaBoost', AdaBoostClassifier()),
        ('GradientBoosting', GradientBoostingClassifier()),
        ('RandomForest', RandomForestClassifier()),
        ('ExtraTrees', ExtraTreesClassifier()),
        ('DecisionTree', DecisionTreeClassifier()),
        ('HistGradientBoosting', HistGradientBoostingClassifier()),
        ('Bagging', BaggingClassifier()),
        ('KNeighbors', KNeighborsClassifier()),
        ('SVM', svm.SVC()),
        ('MLP', MLPClassifier(max_iter=500)),
        ('GaussianNB', GaussianNB()),
        ('LinearRegression', LinearRegression())
    ]

    # Initialize data structures
    node_predictions = pd.DataFrame()
    node_test_predictions = pd.DataFrame()

    # Setup cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Metrics storage
    performance_metrics = pd.DataFrame(columns=['Node', 'BalancedAccuracy', 'TPR', 'TNR'])

    # Store predictions for each fold
    predictions_dict = {name: [] for name, _ in classifiers}
    validation_data = []
    validation_labels = []

    # Performance scores storage
    tpr_scores = []
    tnr_scores = []
    balanced_acc_scores = []

    # Cross-validation training
    print("Starting cross-validation training...")

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        print(f"Fold {fold_num + 1}/5: Training {len(classifiers)} models...")

        for model_idx, (model_name, classifier) in enumerate(classifiers):
            # Train model
            classifier.fit(X_fold_train, y_fold_train)

            # Predict on validation set
            y_val_pred = classifier.predict(X_fold_val)

            # Convert LinearRegression output to binary classification
            if isinstance(classifier, LinearRegression):
                y_val_pred = (y_val_pred > 0.5).astype(int)

            # Calculate performance metrics
            balanced_acc = balanced_accuracy_score(y_fold_val, y_val_pred)
            tn, fp, fn, tp = confusion_matrix(y_fold_val, y_val_pred).ravel()
            tpr = tp / (tp + fn)  # True Positive Rate
            tnr = tn / (fp + tn)  # True Negative Rate

            # Store scores
            tpr_scores.append((model_idx, tpr))
            tnr_scores.append((model_idx, tnr))
            balanced_acc_scores.append((model_idx, balanced_acc))

            # Store predictions
            predictions_dict[model_name].append(y_val_pred)

        # Store validation data
        validation_data.append(X_fold_val)
        validation_labels.append(y_fold_val)

    # Generate test set predictions
    print("Generating test set predictions...")

    for model_name, classifier in classifiers:
        test_pred = classifier.predict(X_test)
        test_pred_df = pd.DataFrame(test_pred, columns=[model_name])
        node_test_predictions = pd.concat([node_test_predictions, test_pred_df], axis=1)

    # Calculate average metrics for each model
    avg_tpr = defaultdict(list)
    avg_tnr = defaultdict(list)
    avg_balanced_acc = defaultdict(list)

    for idx, tpr in tpr_scores:
        avg_tpr[idx].append(tpr)

    for idx, tnr in tnr_scores:
        avg_tnr[idx].append(tnr)

    for idx, balanced_acc in balanced_acc_scores:
        avg_balanced_acc[idx].append(balanced_acc)

    # Store average metrics in DataFrame
    for idx in range(len(classifiers)):
        mean_tpr = np.mean(avg_tpr[idx]) if idx in avg_tpr else np.nan
        mean_tnr = np.mean(avg_tnr[idx]) if idx in avg_tnr else np.nan
        mean_balanced_acc = np.mean(avg_balanced_acc[idx]) if idx in avg_balanced_acc else np.nan

        performance_metrics.loc[idx] = {
            'Node': idx,
            'BalancedAccuracy': mean_balanced_acc,
            'TPR': mean_tpr,
            'TNR': mean_tnr
        }

    print("Performance metrics calculated:")
    print(performance_metrics)

    # Combine validation predictions
    for model_name, pred_list in predictions_dict.items():
        combined_preds = np.concatenate(pred_list)
        node_predictions[model_name] = combined_preds

    # Combine validation data
    combined_validation_X = pd.concat(validation_data, ignore_index=True)
    combined_validation_y = pd.concat(validation_labels, ignore_index=True)

    # Extract feature importances
    feature_importance_dict = extract_feature_importances(classifiers, X_train)

    # Extract important features for each model
    important_features_train, important_features_test = extract_important_features(
        classifiers, feature_importance_dict, X_train, X_test, top_n=10
    )

    node_count = len(classifiers)

    print("================== Ensemble Base Layer Training Complete ==================")

    return (
        node_predictions,
        node_test_predictions,
        performance_metrics,
        node_count,
        combined_validation_y,
        combined_validation_X,
        feature_importance_dict,
        classifiers,
        important_features_train,
        important_features_test
    )


def extract_feature_importances(classifiers, X_data):
    """
    Extract feature importances from classifiers.
    """
    feature_importance_dict = {}
    feature_counts = defaultdict(int)

    for model_name, classifier in classifiers:
        try:
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                importances = np.abs(classifier.coef_).mean(axis=0)
            else:
                importances = None
        except AttributeError:
            importances = None

        if importances is not None:
            # Get top 10 features
            top_indices = np.argsort(importances)[::-1][:10]
            top_features = X_data.columns[top_indices].tolist()

            for feature in top_features:
                feature_counts[feature] += 1

            feature_importance_dict[model_name] = {
                'indices': top_indices,
                'feature_names': top_features
            }
        else:
            # Use most common features from other models
            common_features = sorted(feature_counts.items(),
                                     key=lambda x: x[1],
                                     reverse=True)[:10]
            common_feature_names = [feature for feature, _ in common_features]
            common_feature_indices = [list(X_data.columns).index(feature)
                                      for feature in common_feature_names]

            feature_importance_dict[model_name] = {
                'indices': common_feature_indices,
                'feature_names': common_feature_names
            }

    return feature_importance_dict


def extract_important_features(classifiers, feature_importance_dict, X_train, X_test, top_n=10):
    """
    Extract important features for training and testing.
    """
    train_important_features = pd.DataFrame()
    test_important_features = pd.DataFrame()

    for model_name, _ in classifiers:
        features_info = feature_importance_dict.get(model_name)

        if features_info:
            # Extract selected features
            train_selected = X_train[features_info['feature_names'][:top_n]]
            test_selected = X_test[features_info['feature_names'][:top_n]]

            # Add unique features to combined datasets
            existing_train_cols = set(train_important_features.columns)
            new_train_cols = [col for col in train_selected.columns
                              if col not in existing_train_cols]

            existing_test_cols = set(test_important_features.columns)
            new_test_cols = [col for col in test_selected.columns
                             if col not in existing_test_cols]

            if new_train_cols:
                train_important_features = pd.concat(
                    [train_important_features, train_selected[new_train_cols]],
                    axis=1
                )

            if new_test_cols:
                test_important_features = pd.concat(
                    [test_important_features, test_selected[new_test_cols]],
                    axis=1
                )

    return train_important_features, test_important_features