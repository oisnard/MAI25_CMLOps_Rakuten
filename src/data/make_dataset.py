import pandas as pd 
import os 
import src.tools.tools as tools 
import src.data.preprocessing as preprocessing
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
from sklearn.model_selection import train_test_split
import logging
import warnings



# Function to get the starting line number for processing the dataset
def get_start_line() -> int:
    """
    Get the starting line number for processing the dataset. 
    It allows to simulate new data coming from the fields
    The last starting line number can be stored in a file data/processed/last_start_line.txt
    
    Returns:
        int: The starting line number for processing the dataset.
    """
    last_start_line_file = tools.LAST_START_LINE_FILENAME
    if os.path.exists(last_start_line_file):
        with open(last_start_line_file, 'r') as f:
            last_start_line = f.read().strip()
            return int(last_start_line) if last_start_line.isdigit() else 0
    return 0  # Default to start from the beginning

def save_start_line(line_number: int):
    """
    Save the current line number to the next start line file.
    
    Args:
        line_number (int): The current line number to save.
    """
    last_start_line_file = tools.LAST_START_LINE_FILENAME
    with open(last_start_line_file, 'w') as f:
        f.write(str(line_number))
    logging.info(f"Last start line saved: {line_number}")


# A retravailler pour prendre au fur et à mesure les données selon le ratio
def get_dataset_from_ratio(X_train: pd.DataFrame, 
                           y_train: pd.DataFrame, 
                           ratio: float = 1.) -> tuple:
    """
    Split the dataset into subset in order to reduce training duration
    
    Args:
        X_train (pd.DataFrame): The features DataFrame.
        y_train (pd.DataFrame): The target DataFrame.
        ratio (float): The ratio of training data to total data.
        
    Returns:
        tuple: A tuple containing the resulting X_train & y_train sets
    """
    if not (0. < ratio <= 1.):
        raise ValueError("Ratio must be between 0 and 1")

    if ratio == 1.0:
        return X_train, y_train

    start_line = get_start_line()
    if start_line >= X_train.shape[0]:
        start_line = 0  # Reset to 0 if the start line is greater than the number of rows in the DataFrame
    logging.info(f"Starting line for processing: {start_line}")
    next_line = start_line + int(X_train.shape[0] * ratio)  # Start from the next line after the last processed line
    if next_line > X_train.shape[0]:
        next_line = X_train.shape[0]  # Ensure we don't exceed the DataFrame length
    logging.info(f"Next line to process: {next_line}")

    # Save the current line number for the next run
    save_start_line(next_line)
    # Return the subset of the dataset based on the ratio
    if start_line >= X_train.shape[0]:
        logging.warning("Start line is greater than or equal to the number of rows in the DataFrame. Returning empty DataFrame.")
        return pd.DataFrame(), pd.DataFrame()
    return X_train.iloc[start_line:next_line], y_train.iloc[start_line:next_line]




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load dataset parameters from YAML file
    dataset_params = tools.load_dataset_params_from_yaml()
    logging.info(f"Dataset parameters loaded: {dataset_params}")

    # Process the  raw data file
    logging.info("Starting to process the Y_train raw data file...")
    # Load the Y_train raw data file
    logging.info("Loading Y_train raw data...")
    y_train_df = tools.load_ytrain_raw_data()    

    # Get the mapping dictionary for prdtypecode
    logging.info("Creating mapping dictionary for prdtypecode...")
    mapping_dict = preprocessing.get_mapping_dict(y_train_df)
    logging.info(f"Mapping dictionary for prdtypecode created with {len(mapping_dict)} entries.")
    # Save the mapping dictionary to a pickle file
    tools.save_mapping_dict(mapping_dict, tools.FILE_MAPPING_DICT)
    logging.info(f"Mapping dictionary saved to {tools.FILE_MAPPING_DICT}")


    # Load the X_train raw data file
    logging.info("Loading X_train raw data...")
    x_train_df = tools.load_xtrain_raw_data()

    # Extract the ratio from the dataset parameters
    ratio = dataset_params.get('data_ratio', 1.0)
    if not (0. < ratio <= 1.):
        logging.warning(f"Invalid ratio {ratio} found in dataset parameters. Using default value of 1.0.")
        ratio = 1.0
    logging.info(f"Using ratio {ratio} for extracting training data.")

    # Get the subset of the dataset based on the ratio
    x_train_df, y_train_df = get_dataset_from_ratio(x_train_df, y_train_df, ratio)

    if x_train_df.empty or y_train_df.empty:
        logging.error("X_train or Y_train is empty after applying the ratio. All data have been processed")
        exit(1)

    # Process the X_train raw data file
    logging.info("Starting to process the X_train raw data file...")
    # Process the raw data
    logging.info("Processing X_train raw data...")
    processed_x_train_df = preprocessing.process_raw_data(x_train_df, train_data=True)

    # Process the raw data
    logging.info("Processing Y_train raw data...")
    processed_y_train_df = preprocessing.process_target_raw_data(y_train_df, mapping_dict)

    logging.info("Split dateset into training, validation, and test sets...")
    # Get the split parameters from the dataset parameters
    split_params = dataset_params.get('data_split', {})
    if not split_params:
        logging.ERROR("No split parameters found in dataset parameters. Using default values.")
        exit(1)
    # Split the dataset into training, validation, and test sets

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.get_dataset_from_split(
        processed_x_train_df, 
        processed_y_train_df, 
        split_params
    )
    logging.info(f"Dataset split into training, validation, and test sets with sizes: "
                 f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}, "
                 f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")
    # Save the processed data to CSV files
    

    # Save the dataset to CSV files
    logging.info("Saving processed X_train data to CSV file...")
    os.makedirs(tools.DATA_PROCESSED_DIR, exist_ok=True)
    target_filename = os.path.join(tools.DATA_PROCESSED_DIR, "X_train.csv")
    X_train.to_csv(target_filename, index=True)
    # Save the processed X_val data to a CSV file
    logging.info("Saving processed X_val data to CSV file...")
    target_filename = os.path.join(tools.DATA_PROCESSED_DIR, "X_val.csv")
    X_val.to_csv(target_filename, index=True)
    logging.info(f"Processed X_val data saved to {target_filename}")
    # Save the processed X_test data to a CSV file
    logging.info("Saving processed X_test data to CSV file...")
    target_filename = os.path.join(tools.DATA_PROCESSED_DIR, "X_test.csv")
    X_test.to_csv(target_filename, index=True)
    logging.info(f"Processed X_test data saved to {target_filename}")
    # Save the processed y_train data to a CSV file
    logging.info("Saving processed y_train data to CSV file...")
    target_filename = os.path.join(tools.DATA_PROCESSED_DIR, "y_train.csv")
    y_train.to_csv(target_filename, index=True)
    logging.info(f"Processed y_train data saved to {target_filename}")
    # Save the processed y_val data to a CSV file
    logging.info("Saving processed y_val data to CSV file...")
    target_filename = os.path.join(tools.DATA_PROCESSED_DIR, "y_val.csv")
    y_val.to_csv(target_filename, index=True)
    logging.info(f"Processed y_val data saved to {target_filename}")
    # Save the processed y_test data to a CSV file
    logging.info("Saving processed y_test data to CSV file...")
    target_filename = os.path.join(tools.DATA_PROCESSED_DIR, "y_test.csv")
    y_test.to_csv(target_filename, index=True)
    logging.info(f"Processed y_test data saved to {target_filename}")
    logging.info("Dataset processing completed successfully.")
    logging.info("All processed data saved to the processed data directory.")
