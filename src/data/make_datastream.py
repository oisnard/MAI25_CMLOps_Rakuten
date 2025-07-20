# Since available data is limited to the ENS Data Challenge, we split the data into several parts to simulate a real-world scenario.
# A baseline dataset is created with a ratio defined in the params.yaml file (baseline_ratio parameter).
# The baseline dataset is used for training and evaluation, while the remaining data is used for data drift monitoring and testing.

import pandas as pd 
import os 
import src.data.preprocessing as preprocessing
import src.tools.tools as tools 
import logging
import warnings
import shutil

def make_datastreams(X: pd.DataFrame, y:pd.DataFrame):
    """
    Create datastreams from the available data and save them to the DATA_RAW_STREAM_DIR directory.
    The datastreams are created based on the baseline_ratio defined in params.yaml.
    The baseline dataset is created with a ratio defined in the params.yaml file (baseline_ratio parameter).
    The baseline dataset is used for training and evaluation and data reference, while the remaining data is used for data drift monitoring and testing.
    Args:
        X (DataFrame): The DataFrame containing features.
        y (DataFrame): The DataFrame containing labels.
    Returns:
        baseline_size (int): The size of the baseline dataset.
        datastreams (list): A list of tuples, each containing a DataFrame of features and a DataFrame of labels for each datastream.
    """
    # Load dataset parameters from YAML file
    dataset_params = tools.load_dataset_params_from_yaml()
    if dataset_params is None:
        logging.error("Failed to load dataset parameters from YAML file.")
        raise ValueError("Failed to load dataset parameters from YAML file.")
    baseline_ratio = dataset_params.get('baseline_ratio')
    nb_datastreams = dataset_params.get('nb_datastreams')
    if baseline_ratio is None or nb_datastreams is None:
        logging.error("Missing required parameters in dataset parameters: 'baseline_ratio' or 'nb_datastreams'.")
        raise ValueError("Missing required parameters in dataset parameters: 'baseline_ratio' or 'nb_datastreams'.")
    if not isinstance(baseline_ratio, float) or not (0 < baseline_ratio < 1):
        logging.error(f"Invalid baseline_ratio value: {baseline_ratio}. It should be a float between 0 and 1.")
        raise ValueError(f"Invalid baseline_ratio value: {baseline_ratio}. It should be a float between 0 and 1.")
    if not isinstance(nb_datastreams, int) or nb_datastreams <= 0:
        logging.error(f"Invalid nb_datastreams value: {nb_datastreams}. It should be a positive integer.")
        raise ValueError(f"Invalid nb_datastreams value: {nb_datastreams}. It should be a positive integer.")

    # Create baseline dataset
    baseline_size = int(len(X) * baseline_ratio)
    X_baseline = X.iloc[:baseline_size]
    y_baseline = y.iloc[:baseline_size]
    logging.info(f"Baseline dataset created with size: {len(X_baseline)}")
    # Save baseline dataset
    if not os.path.exists(tools.DATA_RAW_STREAM_DIR):
        os.makedirs(tools.DATA_RAW_STREAM_DIR)
    logging.info(f"Saving baseline dataset to {tools.DATA_RAW_STREAM_DIR}")
    X_baseline.to_csv(os.path.join(tools.DATA_RAW_STREAM_DIR, "X_baseline.csv"), index=True)
    y_baseline.to_csv(os.path.join(tools.DATA_RAW_STREAM_DIR, "y_baseline.csv"), index=True)
    logging.info("Baseline dataset saved.")

    # Create datastreams
    for i in range(nb_datastreams):
        start_index = baseline_size + i * (len(X) - baseline_size) // nb_datastreams
        end_index = start_index + (len(X) - baseline_size) // nb_datastreams
        if end_index > len(X):
            end_index = len(X)
        X_stream = X.iloc[start_index:end_index]
        y_stream = y.iloc[start_index:end_index]
        logging.info(f"Datastream {i} created with size: {X_stream.shape[0]}")
        # Save datastream
        X_stream.to_csv(os.path.join(tools.DATA_RAW_STREAM_DIR, f"X_stream_{i}.csv"), index=True)
        y_stream.to_csv(os.path.join(tools.DATA_RAW_STREAM_DIR, f"y_stream_{i}.csv"), index=True)
        logging.info(f"Datastream {i} saved to {tools.DATA_RAW_STREAM_DIR}")
    logging.info("All datastreams created and saved successfully.")

def check_datastreams():
    """
    Check if the datastreams are created and saved correctly.
    returns:
        bool: True if datastreams are found, False otherwise.
    """
    if not os.path.exists(tools.DATA_RAW_STREAM_DIR):
        if os.path.exists(tools.DATA_PROCESSED_STREAM_DIR):
            shutil.rmtree(tools.DATA_PROCESSED_STREAM_DIR)
        return False
    logging.info(f"Checking for datastream files in {tools.DATA_RAW_STREAM_DIR}")
    files = os.listdir(tools.DATA_RAW_STREAM_DIR)
    if not files:
        logging.info("No datastream files found in the raw stream directory.")
        if os.path.exists(tools.DATA_PROCESSED_STREAM_DIR):
            shutil.rmtree(tools.DATA_PROCESSED_STREAM_DIR)
        return False

    logging.info(f"Datastream files found: {files}")
    return True


   


def main():
    """
    Main function to create datastreams from the available data.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if check_datastreams():
        logger.info("Datastreams already exist. Skipping creation.")
    else:    
        logger.info("Loading dataset for datastream creation...")
        # Load the dataset
        try:
            X = tools.load_xtrain_raw_data()
            y = tools.load_ytrain_raw_data()
        except FileNotFoundError as e:
            raise ValueError(f"Error occured when loading raw data: {e}")

        # Check if the dataset is loaded correctly
        if X.empty or y.empty:
            raise ValueError("The dataset is empty. Please check the input files.")

        # Create datastreams
        make_datastreams(X, y)  

        # Get the mapping dictionary for prdtypecode from the whole target dataset
        logging.info("Creating mapping dictionary for prdtypecode from the whole target dataset...")
        mapping_dict = preprocessing.get_mapping_dict(y)
        # Save the mapping dictionary to a pickle file
        if mapping_dict is None:
            logging.error("Failed to create mapping dictionary.")
            raise ValueError("Failed to create mapping dictionary.")
        logging.info("Saving mapping dictionary to file...")
        # Save the mapping dictionary to a file
        logging.info(f"Saving mapping dictionary to {tools.FILE_MAPPING_DICT}")
        if not isinstance(mapping_dict, dict):
            logging.error("Mapping dictionary is not a valid dictionary.")
            raise ValueError("Mapping dictionary is not a valid dictionary.")
        if not mapping_dict:
            logging.error("Mapping dictionary is empty.")
            raise ValueError("Mapping dictionary is empty.")
        filepath_mapping_dict = tools.save_mapping_dict(mapping_dict, tools.FILE_MAPPING_DICT)
        if not filepath_mapping_dict:
            logging.error("Failed to save mapping dictionary.")
            raise ValueError("Failed to save mapping dictionary.")

if __name__ == "__main__":
    main()