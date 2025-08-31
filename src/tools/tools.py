from dotenv import find_dotenv, load_dotenv
import os
import logging
import pandas as pd
import json
import yaml 
 

# Load environment variables from .env file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Loading environment variables from .env file")
load_dotenv(find_dotenv())  # Automatically find and load the .env file
ENSDATA_LOGIN = os.getenv("ENSDATA_LOGIN")
ENSDATA_PASSWORD = os.getenv("ENSDATA_PASSWORD")

DATA_RAW_DIR = os.getenv("DATA_RAW_DIR")
DATA_RAW_IMAGES_TRAIN_DIR = os.getenv("DATA_RAW_IMAGES_TRAIN_DIR")
DATA_RAW_IMAGES_TEST_DIR = os.getenv("DATA_RAW_IMAGES_TEST_DIR")

DATA_PROCESSED_DIR  = os.getenv("DATA_PROCESSED_DIR")
DATA_RAW_STREAM_DIR = os.getenv("DATA_RAW_STREAM_DIR")  # Directory to save raw datastreams
DATA_PROCESSED_STREAM_DIR = os.getenv("DATA_PROCESSED_STREAM_DIR")  # Directory to save processed datastreams

DATA_MONITORING_SAMPLE_DIR = os.getenv("DATA_MONITORING_SAMPLE_DIR")  # Directory to save monitoring samples

DATA_VIZ_DIR = os.getenv("DATA_VIZ_DIR")  # Directory to save visualization assets

MODEL_DIR = os.getenv("MODEL_DIR")
LOGS_DIR = os.getenv("LOGS_DIR")
METRICS_DIR = os.getenv("METRICS_DIR")  # Directory to save classification reports

FILE_MAPPING_DICT = "mapping_dict.json"  # File to save the mapping dictionary


NUM_CLASSES = 27  # Number of classes for the classification task


# Ensure that the required environment variables are set
logging.info("Checking environment variables...")

# Check if the required directories are set
if not DATA_RAW_DIR or not DATA_PROCESSED_DIR or not MODEL_DIR or not LOGS_DIR:
    logging.error("One or more required directories are not set in the .env file.")
    raise ValueError("Missing environment variables: DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, or LOGS_DIR")

if not DATA_RAW_IMAGES_TRAIN_DIR or not DATA_RAW_IMAGES_TEST_DIR:
    logging.error("DATA_RAW_IMAGES_TRAIN_DIR and DATA_RAW_IMAGES_TEST_DIR must be set in the .env file.")
    raise ValueError("Missing environment variables: DATA_RAW_IMAGES_TRAIN_DIR or DATA_RAW_IMAGES_TEST_DIR")

if not MODEL_DIR or not LOGS_DIR:
    logging.error("MODEL_DIR and LOGS_DIR must be set in the .env file.")
    raise ValueError("Missing environment variables: MODEL_DIR or LOGS_DIR")

# Check if the login and password are set
ENSDATA_LOGIN = ENSDATA_LOGIN.strip() if ENSDATA_LOGIN else None
ENSDATA_PASSWORD = ENSDATA_PASSWORD.strip() if ENSDATA_PASSWORD else None


# Ensure that the login and password are not empty
if not ENSDATA_LOGIN:
    logging.error("ENSDATA_LOGIN must be set in the .env file.")
    raise ValueError("Missing environment variable: ENSDATA_LOGIN")

if not ENSDATA_PASSWORD:
    logging.error("ENSDATA_PASSWORD must be set in the .env file.")
    raise ValueError("Missing environment variable: ENSDATA_PASSWORD")


logging.info("Environment variables loaded successfully.")
    
X_TRAIN_RAW_FILENAME = "X_train_update.csv"
X_TEST_RAW_FILENAME = "X_test_update.csv"
Y_TRAIN_RAW_FILENAME = "Y_train_CVw08PX.csv"

LAST_START_LINE_FILENAME = os.path.join(DATA_PROCESSED_DIR, "last_start_line.txt")

def load_xtrain_raw_data():
    """
    Load the X_train raw data from the specified directory.
    """
    x_train_path = os.path.join(DATA_RAW_DIR, X_TRAIN_RAW_FILENAME)
    if not os.path.exists(x_train_path):
        logging.error(f"X_train raw data file not found at {x_train_path}")
        raise FileNotFoundError(f"X_train raw data file not found at {x_train_path}")
    df = pd.read_csv(x_train_path, index_col=0)
    return df

def load_xtest_raw_data():
    """
    Load the X_test raw data from the specified directory.
    """
    x_test_path = os.path.join(DATA_RAW_DIR, X_TEST_RAW_FILENAME)
    if not os.path.exists(x_test_path):
        logging.error(f"X_test raw data file not found at {x_test_path}")
        raise FileNotFoundError(f"X_test raw data file not found at {x_test_path}")
    df = pd.read_csv(x_test_path, index_col=0)
    return df


def load_ytrain_raw_data():
    """
    Load the Y_train raw data from the specified directory.
    """
    y_train_path = os.path.join(DATA_RAW_DIR, Y_TRAIN_RAW_FILENAME)
    if not os.path.exists(y_train_path):
        logging.error(f"Y_train raw data file not found at {y_train_path}")
        raise FileNotFoundError(f"Y_train raw data file not found at {y_train_path}")
    df = pd.read_csv(y_train_path, index_col=0)
    return df



def get_filepath_test(product_id, image_id):
    """
    Function to return the filepath of an image of X_test based on its product_id and image_id 
    Assumpion : all test images are stored in the filepath C:\\Rakuten\\images\\image_test\\
    """
    filename = "image_" + str(image_id) + "_product_" + str(product_id) + ".jpg"
    filepath = os.path.join(DATA_RAW_IMAGES_TEST_DIR, filename)
    
    return filepath

def get_filepath_train(product_id, image_id):
    """ 
    Function to return the filepath of an image based on its product_id and image_id 
    Assumpion : all train images are stored in the filepath C:\\Rakuten\\images\\image_train\\
    """    
    filename = "image_" + str(image_id) + "_product_" + str(product_id) + ".jpg"
    filepath = os.path.join(DATA_RAW_IMAGES_TRAIN_DIR, filename)
    
    return filepath

# To save mapping of prdtypecode to integer between 0 and 26
def save_mapping_dict(mapping_dict, filename=FILE_MAPPING_DICT):
    """
    Save the mapping dictionary to a json file.
    """
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    filepath = os.path.join(DATA_PROCESSED_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, ensure_ascii=False, indent=4)
    logging.info(f"Mapping dictionary saved to {filepath}")
    return filepath

def load_mapping_dict(filename=FILE_MAPPING_DICT):
    """
    Load the mapping dictionary from a json file.
    { prdtypecode : integer (between 0 and 26)}
    """
    filepath = os.path.join(DATA_PROCESSED_DIR, filename)
    if not os.path.exists(filepath):
        logging.error(f"Mapping dictionary file not found at {filepath}")
        raise FileNotFoundError(f"Mapping dictionary file not found at {filepath}")
    with open(filepath, 'r') as f:
        mapping_dict = json.load(f)
    logging.info(f"Mapping dictionary loaded from {filepath}")

    # Convert keys and values to integers
    results = {int(k): int(v) for k, v in mapping_dict.items()}
    return results 


def load_reverse_mapping_dict(filename=FILE_MAPPING_DICT):
    """
    Load the reverse mapping dictionary from a json file.
    {integer (between 0 and 26) : prdtypecode}

    """
    filepath = os.path.join(DATA_PROCESSED_DIR, filename)
    if not os.path.exists(filepath):
        logging.error(f"Mapping dictionary file not found at {filepath}")
        raise FileNotFoundError(f"Mapping dictionary file not found at {filepath}")
    with open(filepath, 'r') as f:
        mapping_dict = json.load(f)
    logging.info(f"Mapping dictionary loaded from {filepath}")

    # Convert keys and values to integers
    results = {int(v): int(k) for k, v in mapping_dict.items()}
    return results 


def load_dataset_params_from_yaml(file_path: str="params.yaml") -> dict:
    """
    Load dataset parameters from a YAML file.
    
    Args:
        file_path (str): The path to the YAML file.
        
    Returns:
        dict: A dictionary containing the dataset parameters.
    """
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params   


def load_datasets() -> tuple:
    """
    Function to load the datasets for training and validation.
    Returns:
        X_train, y_train, X_val, y_val : DataFrames containing the training and validation data.
    """
    try:
        X_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"), index_col=0)
        y_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv"), index_col=0)
        X_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_val.csv"), index_col=0)
        y_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_val.csv"), index_col=0)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading datasets: {e}")
        raise
    # Ensure that the datasets are not empty
    if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
        logging.error("One or more datasets are empty.")
        raise ValueError("One or more datasets are empty.")
    # Ensure that the indices of X_train and y_train match
    if not X_train.index.equals(y_train.index):
        logging.error("Indices of X_train and y_train do not match.")
        raise ValueError("Indices of X_train and y_train do not match.")
    if not X_val.index.equals(y_val.index):
        logging.error("Indices of X_val and y_val do not match.")
        raise ValueError("Indices of X_val and y_val do not match.")
    # Ensure that the 'feature' column exists in X_train, X_val, and X_test
    if 'image_path' not in X_train.columns or 'image_path' not in X_val.columns:
        logging.error("The 'image_path' column is missing in one of the datasets.")
        raise ValueError("The 'image_path' column is missing in one of the datasets.")
    # Ensure that the 'prdtypecode' column exists in y_train, y_val, and y_test
    if 'prdtypecode' not in y_train.columns or 'prdtypecode' not in y_val.columns:
        logging.error("The 'prdtypecode' column is missing in one of the label datasets.")
        raise ValueError("The 'prdtypecode' column is missing in one of the label datasets.")
    # Ensure that the 'prdtypecode' column is of integer type
    if not pd.api.types.is_integer_dtype(y_train['prdtypecode']) or not pd.api.types.is_integer_dtype(y_val['prdtypecode']):
        logging.error("The 'prdtypecode' column must be of integer type.")
        raise ValueError("The 'prdtypecode' column must be of integer type.")
    # Ensure that the 'image_path' column is of string type
    if not pd.api.types.is_string_dtype(X_train['image_path']) or not pd.api.types.is_string_dtype(X_val['feature']):
        logging.error("The 'image_path' column must be of string type.")
        raise ValueError("The 'image_path' column must be of string type.")
    # Ensure that the 'prdtypecode' column is not empty
    if y_train['prdtypecode'].isnull().any() or y_val['prdtypecode'].isnull().any():
        logging.error("The 'prdtypecode' column contains null values.")
        raise ValueError("The 'prdtypecode' column contains null values.")
    # Ensure that the 'feature' column is not empty
    if X_train['image_path'].isnull().any() or X_val['image_path'].isnull().any():
        logging.error("The 'image_path' column contains null values.")
        raise ValueError("The 'image_path' column contains null values.")
    return X_train, X_val, y_train, y_val


def load_full_datasets() -> tuple:
    """
    Function to load the full dataset for training and validation.
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test : DataFrames containing the training, validation, and test data.
    """
    try:
        X_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"), index_col=0)
        y_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv"), index_col=0)
        X_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_val.csv"), index_col=0)
        y_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_val.csv"), index_col=0)
        X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"), index_col=0)
        y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv"), index_col=0)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading datasets: {e}")
        raise
    # Ensure that the datasets are not empty
    if X_train.empty or y_train.empty or X_val.empty or y_val.empty or X_test.empty or y_test.empty:
        logging.error("One or more datasets are empty.")
        raise ValueError("One or more datasets are empty.")
    # Ensure that the indices of X_train and y_train match
    if not X_train.index.equals(y_train.index):
        logging.error("Indices of X_train and y_train do not match.")
        raise ValueError("Indices of X_train and y_train do not match.")
    if not X_val.index.equals(y_val.index):
        logging.error("Indices of X_val and y_val do not match.")
        raise ValueError("Indices of X_val and y_val do not match.")
    if not X_test.index.equals(y_test.index):
        logging.error("Indices of X_test and y_test do not match.")
        raise ValueError("Indices of X_test and y_test do not match.")
    # Ensure that the 'feature' column exists in X_train, X_val, and X_test
    if 'image_path' not in X_train.columns or 'image_path' not in X_val.columns or 'image_path' not in X_test.columns:
        logging.error("The 'image_path' column is missing in one of the datasets.")
        raise ValueError("The 'image_path' column is missing in one of the datasets.")
    # Ensure that the 'prdtypecode' column exists in y_train, y_val, and y_test
    if 'prdtypecode' not in y_train.columns or 'prdtypecode' not in y_val.columns or 'prdtypecode' not in y_test.columns:
        logging.error("The 'prdtypecode' column is missing in one of the label datasets.")
        raise ValueError("The 'prdtypecode' column is missing in one of the label datasets.")
    # Ensure that the 'prdtypecode' column is of integer type
    if not pd.api.types.is_integer_dtype(y_train['prdtypecode']) or not pd.api.types.is_integer_dtype(y_val['prdtypecode']) or not pd.api.types.is_integer_dtype(y_test['prdtypecode']):
        logging.error("The 'prdtypecode' column must be of integer type.")
        raise ValueError("The 'prdtypecode' column must be of integer type.")
    # Ensure that the 'image_path' column is of string type
    if not pd.api.types.is_string_dtype(X_train['image_path']) or not pd.api.types.is_string_dtype(X_val['feature']) or not pd.api.types.is_string_dtype(X_test['feature']):
        logging.error("The 'image_path' column must be of string type.")
        raise ValueError("The 'image_path' column must be of string type.")
    # Ensure that the 'prdtypecode' column is not empty
    if y_train['prdtypecode'].isnull().any() or y_val['prdtypecode'].isnull().any() or y_test['prdtypecode'].isnull().any():
        logging.error("The 'prdtypecode' column contains null values.")
        raise ValueError("The 'prdtypecode' column contains null values.")
    # Ensure that the 'feature' column is not empty
    if X_train['image_path'].isnull().any() or X_val['image_path'].isnull().any() or X_test['image_path'].isnull().any():
        logging.error("The 'image_path' column contains null values.")
        raise ValueError("The 'image_path' column contains null values.")
    return X_train, X_val, y_train, y_val, X_test, y_test


def load_test_datasets() -> tuple:
    """
    Function to load the test dataset.
    Returns:
        X_test, y_test : DataFrames containing the test data.
    """
    try:
        X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"), index_col=0)
        y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv"), index_col=0)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading datasets: {e}")
        raise
    # Ensure that the datasets are not empty
    if X_test.empty or y_test.empty:
        logging.error("One or more datasets are empty.")
        raise ValueError("One or more datasets are empty.")
    # Ensure that the indices of X_test and y_test match
    if not X_test.index.equals(y_test.index):
        logging.error("Indices of X_test and y_test do not match.")
        raise ValueError("Indices of X_test and y_test do not match.")
    # Ensure that the 'feature' column exists in X_test
    if 'image_path' not in X_test.columns:
        logging.error("The 'image_path' column is missing in the test dataset.")
        raise ValueError("The 'image_path' column is missing in the test dataset.")
    # Ensure that the 'prdtypecode' column exists in y_test
    if 'prdtypecode' not in y_test.columns:
        logging.error("The 'prdtypecode' column is missing in the test label dataset.")
        raise ValueError("The 'prdtypecode' column is missing in the test label dataset.")
    # Ensure that the 'prdtypecode' column is of integer type
    if not pd.api.types.is_integer_dtype(y_test['prdtypecode']):
        logging.error("The 'prdtypecode' column must be of integer type.")
        raise ValueError("The 'prdtypecode' column must be of integer type.")
    # Ensure that the 'image_path' column is of string type
    if not pd.api.types.is_string_dtype(X_test['image_path']):
        logging.error("The 'image_path' column must be of string type.")
        raise ValueError("The 'image_path' column must be of string type.")
    # Ensure that the 'prdtypecode'
    return X_test, y_test