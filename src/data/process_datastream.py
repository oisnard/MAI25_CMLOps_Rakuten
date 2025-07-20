# Since available data is limited to the ENS Data Challenge, we split the data into several parts to simulate a real-world scenario.
# A baseline dataset is created with a ratio defined in the params.yaml file (baseline_ratio parameter).
# The baseline dataset is used for training and evaluation, while the remaining data is used for data drift monitoring and testing.
# This code is part of the data processing pipeline.
# It processes the data to create cleaned and structured datastream for model training and evaluation.

import pandas as pd 
import os 
import src.tools.tools as tools 
import logging
import warnings
import shutil
import src.data.preprocessing as preprocessing
from sklearn.model_selection import train_test_split


def get_next_stream() -> tuple:
    """
    Retrieve the next datastream from the available data.
    This function loads the next datastream files from the DATA_RAW_STREAM_DIR directory.
    It returns the DataFrames for features (X) and labels (y), and a boolean indicating if the stream is a baseline stream.
    The next datastream is determined by the files available in the DATA_RAW_STREAM_DIR directory.
    If no files are found, it raises an error.
    Returns:
        tuple: A tuple containing:
            - X (DataFrame): The DataFrame containing features for the next datastream.
            - y (DataFrame): The DataFrame containing labels for the next datastream.
            - baseline_stream (bool): A boolean indicating if the datastream is a baseline stream.
            - stream_id (str): The ID of the datastream (e.g., "stream_0").
    Raises:
        ValueError: If no datastream files are found in the DATA_RAW_STREAM_DIR directory or if the datastream files are empty.
    """
    # Implement the logic to process the next datastream
    X_files = sorted(f for f in os.listdir(tools.DATA_RAW_STREAM_DIR) if f.startswith("X_") and f.endswith(".csv"))
    y_files = sorted(f for f in os.listdir(tools.DATA_RAW_STREAM_DIR) if f.startswith("y_") and f.endswith(".csv"))
    if not X_files or not y_files:
        logging.error("No datastream files found in the raw stream directory.")
        raise ValueError("No datastream files found in the raw stream directory.")
    # Load the next datastream files
    logging.info(f"Loading next datastream files: {X_files[0]}, {y_files[0]}")
    print(os.path.join(tools.DATA_RAW_STREAM_DIR, X_files[0]))
    X = pd.read_csv(os.path.join(tools.DATA_RAW_STREAM_DIR, X_files[0]), index_col=0)
    y = pd.read_csv(os.path.join(tools.DATA_RAW_STREAM_DIR, y_files[0]), index_col=0)
    if X.empty or y.empty:
        logging.error("The datastream files are empty.")
        raise ValueError("The datastream files are empty.")
    logging.info(f"Loaded datastream with shapes - X: {X.shape}, y: {y.shape}")
    if "baseline" in X_files[0]:
        if "baseline" in y_files[0]:
            baseline_stream = True
        else:
            logging.error("Baseline stream file mismatch: X and y files do not match.")
            raise ValueError("Baseline stream file mismatch: X and y files do not match.")
    else:
        baseline_stream = False
    # Moving the processed files from the raw stream directory
    logging.info(f"Moving processed files: {X_files[0]}, {y_files[0]} from {tools.DATA_RAW_STREAM_DIR} to {tools.DATA_PROCESSED_STREAM_DIR} ")
    if not os.path.exists(tools.DATA_PROCESSED_STREAM_DIR):
        os.makedirs(tools.DATA_PROCESSED_STREAM_DIR)
    shutil.move(os.path.join(tools.DATA_RAW_STREAM_DIR, X_files[0]), os.path.join(tools.DATA_PROCESSED_STREAM_DIR, X_files[0]))
    shutil.move(os.path.join(tools.DATA_RAW_STREAM_DIR, y_files[0]), os.path.join(tools.DATA_PROCESSED_STREAM_DIR, y_files[0]))
    return X, y, baseline_stream, X_files[0].split('_', 1)[1].rsplit('.', 1)[0]

def process_next_datastream(X:pd.DataFrame, y:pd.DataFrame, baseline_stream:bool, stream_name:str) -> None:
    """
    Process the next datastream.
    This function is a placeholder for processing logic that would be applied to the next datastream.
    Args:
        X (DataFrame): The DataFrame containing features for the next datastream.
        y (DataFrame): The DataFrame containing labels for the next datastream.
        baseline_stream (bool): A boolean indicating if the datastream is a baseline stream.
    """
    # Placeholder for processing logic
    logging.info(f"Processing datastream - Stream: {stream_name}, Shapes - X: {X.shape}, y: {y.shape}")
    # Process the raw data
    X_processed = preprocessing.process_raw_data(X, train_data=not baseline_stream)
    logging.info(f"Processed X DataFrame shape: {X_processed.shape}")
    # Process the target raw data   
    mapping_dict = tools.load_mapping_dict()
    if mapping_dict is None:
        logging.error("Failed to get the mapping dictionary for prdtypecode.")
        raise ValueError("Failed to get the mapping dictionary for prdtypecode.")
    y_processed = preprocessing.process_target_raw_data(y, mapping_dict)
    logging.info(f"Processed y DataFrame shape: {y_processed.shape}")

    # Load dataset parameters from YAML file
    dataset_params = tools.load_dataset_params_from_yaml()
    logging.info(f"Dataset parameters loaded: {dataset_params}")
    split_params = dataset_params.get('data_split', {})
    if not split_params:
        logging.ERROR("No split parameters found in dataset parameters. Using default values.")
        exit(1)
    # Split the dataset into training, validation, and test sets
    logging.info("Split the dataset into training, validation and test sets")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.get_dataset_from_split(
        X_processed, 
        y_processed, 
        split_params
    )   

    # Save the processed DataFrames to CSV files
    logging.info(f"Saving processed DataFrames to CSV files in {tools.DATA_PROCESSED_DIR}")
    if not os.path.exists(tools.DATA_PROCESSED_DIR):
        os.makedirs(tools.DATA_PROCESSED_DIR)
    X_train.to_csv(os.path.join(tools.DATA_PROCESSED_DIR, f"X_train.csv"), index=True)
    X_val.to_csv(os.path.join(tools.DATA_PROCESSED_DIR, f"X_val.csv"), index=True)
    X_test.to_csv(os.path.join(tools.DATA_PROCESSED_DIR, f"X_test.csv"), index=True)
    y_train.to_csv(os.path.join(tools.DATA_PROCESSED_DIR, f"y_train.csv"), index=True)
    y_val.to_csv(os.path.join(tools.DATA_PROCESSED_DIR, f"y_val.csv"), index=True)
    y_test.to_csv(os.path.join(tools.DATA_PROCESSED_DIR, f"y_test.csv"), index=True)

    # Log the completion of the datastream processing
    logging.info(f"Completed processing for datastream: {stream_name}")


def main():
    """
    Main function to load the next datastream, process it, and save the processed DataFrame.
    """
    logging.basicConfig(level=logging.INFO)

    # Load the next datastream
    X, y, baseline_stream, stream_name = get_next_stream()
    if baseline_stream:
        logging.info("Processing baseline datastream...")
    else:
        logging.info("Processing regular datastream...")
    # Process the next datastream
    process_next_datastream(X, y, baseline_stream, stream_name)

if __name__ == "__main__":
    main()


