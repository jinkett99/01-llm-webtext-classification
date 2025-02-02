import sys
sys.path.append('../src')  # Add the src directory to the system path

import pandas as pd
from file_handling import read_parquet_in_chunks

def limit_records_per_uen(df, max_records=10):
    """
    Limit the number of records per unique UEN to a maximum of `max_records`.
    """
    return df.groupby('UEN').apply(lambda x: x.head(max_records)).reset_index(drop=True)

def merge_with_embedding(astar_df, embedding_df, output_filename):
    """
    Merges the given astar DataFrame (either train or test) with the embedding DataFrame on 'UEN'.
    Drops 'CLUSTER_DEFN' column and saves the result.
    """
    merged_df = pd.merge(astar_df, embedding_df, on='UEN', how='inner')
    merged_df = merged_df.drop(columns=['CLUSTER_DEFN'])
    output_path = f'data/processed/{output_filename}'
    merged_df.to_csv(output_path, index=False)
    return merged_df

if __name__ == "__main__":
    file_path = "data/processed_parquet/embedding_subset.parquet"
    embedding_df = read_parquet_in_chunks(file_path, chunksize=10000)
    
    # Drop unnecessary columns
    cols_to_drop = ['count_0', 'count_1', 'count_2', 'soft_label_category', 'Actual_Category', 'primary_key']
    embedding_df = embedding_df.drop(columns=cols_to_drop)
    
    # Limit records per UEN
    subset_df = limit_records_per_uen(embedding_df)
    
    # Load ASTAR datasets
    astar_train = pd.read_csv("data/raw/astar_train.csv")
    astar_test = pd.read_csv("data/raw/astar_test.csv")
    
    # Merge datasets
    merged_train = merge_with_embedding(astar_train, subset_df, 'train.csv')
    merged_test = merge_with_embedding(astar_test, subset_df, 'test.csv')