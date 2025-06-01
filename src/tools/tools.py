from dotenv import find_dotenv, load_dotenv
import os
import logging
import pandas as pd


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

MODEL_DIR = os.getenv("MODEL_DIR")
LOGS_DIR = os.getenv("LOGS_DIR")

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
logging.info(f"ENSDATA_LOGIN: {ENSDATA_LOGIN}")
logging.info(f"ENSDATA_PASSWORD: {'***' if ENSDATA_PASSWORD else None}")

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