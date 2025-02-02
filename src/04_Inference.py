import pandas as pd
import pickle
import xgboost as xgb

def run_inference_and_group(model_path, dataset_path, output_path, exclude_cols):
    """
    Runs inference on a dataset and processes the results according to specified rules.
    
    Rules:
    1. If at least one UEN is predicted as class 1, output prediction is 1 (Innovative Firm).
    2. Compute the average predicted probability for class 1 across UEN groups.
    
    Parameters:
    - model_path: Path to the pre-trained model.
    - dataset_path: Path to the dataset for inference.
    - output_path: Path where the output dataframe with labels will be saved.
    - exclude_cols: Columns to exclude from the dataset before inference.
    
    Returns:
    - grouped_df: DataFrame with UEN and the determined label + predicted probabilities (class 1).
    """
    # Load the pre-trained model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    
    # Load the dataset for inference
    df = pd.read_csv(dataset_path)
    data = df.drop(columns=exclude_cols, errors='ignore')
    
    # Extract UEN and features
    UEN = data["UEN"]
    X = data.drop(columns=["UEN"], errors='ignore')
    
    # Run inference (0,1 predictions)
    y_pred = model.predict(X)
    
    # Run prediction probabilities for class 1 (innovative)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Create the output dataframe with predictions
    output_df = df.copy()
    output_df["label"] = y_pred
    output_df["predict_proba_1"] = y_pred_proba
    
    # Group by UEN and apply the specified rules
    def determine_label_and_proba(group):
        label = 1 if (group["label"] == 1).any() else 0
        avg_proba = group["predict_proba_1"].mean()
        return pd.Series({"label": label, "avg_predict_proba_1": avg_proba})
    
    grouped_df = output_df.groupby("UEN").apply(determine_label_and_proba).reset_index()
    
    # Save the resulting dataframe to a CSV file
    grouped_df.to_csv(output_path, index=False)
    
    return grouped_df

if __name__ == "__main__":
    model_path = "models/XGBoost_modelv1.pkl"
    dataset_path = "data/processed/embedding_subset.csv"
    output_path = "data/inference_output/prod_labelled_v1.csv"
    exclude_cols = ["CLUSTER_DEFN"]
    
    # Run inference
    grouped_df = run_inference_and_group(model_path, dataset_path, output_path, exclude_cols)
    
    # Print summary
    print(grouped_df.head())
    print("Total records:", len(grouped_df))
    print("Label distribution:")
    print(grouped_df["label"].value_counts())

