import os
import pandas as pd
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient, ContainerClient
from io import BytesIO

def read_parquet_blobs_to_dataframe(connection_string, container_name, folder_names):
    """
    Reads and concatenates Parquet files from Azure Blob Storage into a Pandas DataFrame.
    """
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    dfs = []
    for folder_name in folder_names:
        blob_list = container_client.list_blobs(name_starts_with=folder_name)
        
        for blob in blob_list:
            blob_client = container_client.get_blob_client(blob)
            blob_data = blob_client.download_blob()
            blob_bytes = blob_data.readall()
            
            with BytesIO(blob_bytes) as f:
                df = pd.read_parquet(f)
                dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def subset_by_uen(df, uen_column='UEN', max_records=10):
    """
    Subsets the DataFrame by choosing up to a specified number of random records per unique UEN.
    """
    return df.groupby(uen_column).apply(lambda x: x.sample(n=min(len(x), max_records), random_state=1)).reset_index(drop=True)

def process_embeddings(connection_string, container_name, folder_names, output_csv, subset_output_csv):
    """
    End-to-end function to load, process, and save embedding data.
    """
    # Read Parquet files
    final_df = read_parquet_blobs_to_dataframe(connection_string, container_name, folder_names)
    
    # Save raw embeddings
    final_df.to_csv(output_csv, index=False)
    
    # Drop unwanted columns
    cols_to_drop = ['count_0', 'count_1', 'count_2', 'soft_label_category', 'Actual_Category', 'primary_key']
    df = final_df.drop(columns=cols_to_drop)
    
    # Subset by UEN
    subset_df = subset_by_uen(df)
    
    # Save subset data
    subset_df.to_csv(subset_output_csv, index=False)
    
    return subset_df

if __name__ == "__main__":
    CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=bdamlsa;AccountKey=MdO6w+knH6wyjtpfHjiV0ht6Dl9CVeC/f/XYswzdwXi71Mp4+hfDdAYWKoqsdLkiLl32mdU7hACMluHzMueR4g==;EndpointSuffix=core.windows.net"
    CONTAINER_NAME = "revenue-generating-pred"
    FOLDER_NAMES = ["vectorized_train_data", "vectorized_validate_data", "vectorized_test_data", "vectorized_remaining_data"]
    OUTPUT_CSV = "data/processed/embedding.csv"
    SUBSET_OUTPUT_CSV = "data/processed/embedding_subset.csv"
    
    process_embeddings(CONNECTION_STRING, CONTAINER_NAME, FOLDER_NAMES, OUTPUT_CSV, SUBSET_OUTPUT_CSV)