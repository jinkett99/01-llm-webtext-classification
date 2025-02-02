import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

def train_evaluate_model(model, dataset_path, exclude_columns, target_column, param_grid, model_path, test_set_path):
    """
    Trains a machine learning model with hyperparameter tuning, saves the best model,
    evaluates its performance, and exports predictions on the test set.
    """
    # Load dataset
    data = pd.read_csv(dataset_path)
    UEN = data[exclude_columns]
    X = data.drop(columns=exclude_columns + [target_column])
    y = data[target_column]
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp, UEN_train, UEN_temp = train_test_split(
        X, y, UEN, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, UEN_val, UEN_test = train_test_split(
        X_temp, y_temp, UEN_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Hyperparameter tuning
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, random_state=42)
    random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    print("Best hyperparameters:", random_search.best_params_)
    best_model = random_search.best_estimator_
    
    # Save model
    with open(model_path, "wb") as file:
        pickle.dump(best_model, file)
    
    # Model evaluation
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba)
    }
    print("Evaluation metrics:", metrics)
    
    # Load test set for predictions
    test_df = pd.read_csv(test_set_path)
    X_test_final = test_df.drop(columns=exclude_columns, errors='ignore')
    test_df["predicted_label"] = best_model.predict(X_test_final)
    test_df["predicted_proba"] = best_model.predict_proba(X_test_final)[:, 1]
    
    # Save predictions
    test_df.to_csv("data/processed/test_with_predictions.csv", index=False)
    
    return metrics

if __name__ == "__main__":
    train_file_path = 'data/processed/train.csv'
    test_file_path = 'data/processed/test.csv'
    model_output_path = "models/xgboost_model.pkl"
    
    exclude_cols = ['UEN', 'sector']
    target_col = "label"
    param_grid = {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]}
    
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    train_evaluate_model(xgb_model, train_file_path, exclude_cols, target_col, param_grid, model_output_path, test_file_path)

