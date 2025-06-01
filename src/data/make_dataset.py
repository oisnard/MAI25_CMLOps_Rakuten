import pandas as pd 
import os 
import src.tools.tools as tools 
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
from sklearn.model_selection import train_test_split
import logging
import warnings
import yaml 

# Suppress warnings from BeautifulSoup
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Designation and description columns may contain HTML tags, which need to be removed.
def remove_all_html_tags(sentence: str) -> str:
    """
    Remove all HTML tags from the given text.
    
    Args:
        text (str): The input text containing HTML tags.
        
    Returns:
        str: The text with all HTML tags removed.
    """
    if not isinstance(sentence, str):
        raise ValueError("Input should be a string")

    soup = BeautifulSoup(sentence, 'lxml')
    result = soup.getText('. ', strip=True)
    return result

# Function to process the X_train raw data file
def process_raw_data(df: pd.DataFrame, train_data: bool = True) -> pd.DataFrame:
    """
    Process the X_train raw data file.
    """

    # Check if X_train is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("X_train should be a pandas DataFrame")

    # Check if X_train is not empty
    if df.empty:
        raise ValueError("X_train should not be empty")

    # Check if required columns are present
    required_columns = ['designation', 'description', 'productid', 'imageid']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"X_train should contain '{column}' column")

    # Processes Nan in the 'description' column
    df['description'] = df['description'].fillna(str(""))

    # Remove HTML tags from 'description' and 'designation' columns
    df['description'] = df["description"].apply(remove_all_html_tags)
    df['designation'] = df['designation'].apply(remove_all_html_tags)

    # Create a new 'feature' column by concatenating 'designation' and 'description'. 
    # The separator is ". " since the CamemBERT tokenizer will be used later, and it is better to have a '. ' after the period.
    df['feature'] = df['designation'].astype(str) + str(". ") + df['description'].astype(str)

    # Drop the 'designation' and 'description' columns
    df = df.drop(columns=['designation', 'description'], axis=1)

    # Define the 'filepath' column based on whether it is training data or test data
    if train_data:      
        df['image_path']= df.apply(lambda x : tools.get_filepath_train(x['productid'], x['imageid']), axis = 1)
    else:
        df['image_path']= df.apply(lambda x : tools.get_filepath_test(x['productid'], x['imageid']), axis = 1)

    # Drop the 'productid' and 'imageid' columns
    df = df.drop(columns=['productid', 'imageid'], axis=1)

    # Ensure the 'feature' column is of type string
    df['feature'] = df['feature'].astype(str)
    # Ensure the 'image_path' column is of type string
    df['image_path'] = df['image_path'].astype(str)

    return df


def get_mapping_dict(df: pd.DataFrame) -> dict:
    """
    Create a mapping dictionary from the original prdtypecode to the target prdtypecode.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the 'prdtypecode' column.
        
    Returns:
        dict: A mapping dictionary where keys are original prdtypecodes and values are target prdtypecodes.
    """
    if 'prdtypecode' not in df.columns:
        raise ValueError("DataFrame must contain 'prdtypecode' column")

    list_prdtypecode = df['prdtypecode'].unique()
    target_prdtypecode = range(len(list_prdtypecode))

    # Create a mapping from the original prdtypecode to the target prdtypecode
    mapping_dict = {original: target for original, target in zip(list_prdtypecode, target_prdtypecode)}
    
    return mapping_dict


def process_target_raw_data(df: pd.DataFrame, mapping_dict: dict) -> pd.DataFrame:
    """
    Process the Y_train raw data file.
    """
    # Check if Y_train is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Y_train should be a pandas DataFrame")

    # Check if Y_train is not empty
    if df.empty:
        raise ValueError("Y_train should not be empty")

    # Check if required columns are present
    required_columns = ['prdtypecode']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Y_train should contain '{column}' column")

    # Replace the prdtypecode with the target prdtypecode
    df['prdtypecode'] = df['prdtypecode'].replace(to_replace=mapping_dict)

    # Ensure the 'prdtypecode' column is of type int
    df['prdtypecode'] = df['prdtypecode'].astype(int)


    return df   


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

    train_size = int(len(X_train) * ratio)
    
    return X_train.iloc[:train_size], y_train.iloc[:train_size]


def get_dataset_from_split(X_train: pd.DataFrame,
                           y_train: pd.DataFrame,
                           split_params: dict) -> tuple:
    """
    Split the dataset into training, validation, and test sets based on the provided split parameters.
    
    Args:
        X_train (pd.DataFrame): The features DataFrame.
        y_train (pd.DataFrame): The target DataFrame.
        split_params (dict): The parameters for splitting the dataset.
        
    Returns:
        tuple: A tuple containing the resulting X_train, X_val, X_test, y_train, y_val, y_test sets
    """
    if 'validation_size' not in split_params or 'test_size' not in split_params:
        raise ValueError("Split parameters must contain 'validation_size' and 'test_size'")

    # Split the data into training and temporary sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_train, y_train, 
        test_size=split_params['test_size'], 
        random_state=split_params.get('random_state', 42), 
        shuffle=split_params.get('shuffle', True), 
        stratify=y_train.prdtypecode
    )

    # Split the temporary set into training and validation sets
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_temp, y_temp,
        test_size=split_params['validation_size'],
        random_state=split_params.get('random_state', 42),
        shuffle=split_params.get('shuffle', True),
        stratify=split_params.get('stratify', None)
    )

    return X_train_final, X_val, X_test, y_train_final, y_val, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load dataset parameters from YAML file
    dataset_params = load_dataset_params_from_yaml()
    logging.info(f"Dataset parameters loaded: {dataset_params}")

    # Process the  raw data file
    logging.info("Starting to process the Y_train raw data file...")
    # Load the Y_train raw data file
    logging.info("Loading Y_train raw data...")
    y_train_df = tools.load_ytrain_raw_data()    

    # Get the mapping dictionary for prdtypecode
    logging.info("Creating mapping dictionary for prdtypecode...")
    mapping_dict = get_mapping_dict(y_train_df)
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


    # Process the X_train raw data file
    logging.info("Starting to process the X_train raw data file...")
    # Process the raw data
    logging.info("Processing X_train raw data...")
    processed_x_train_df = process_raw_data(x_train_df, train_data=True)

    # Process the raw data
    logging.info("Processing Y_train raw data...")
    processed_y_train_df = process_target_raw_data(y_train_df, mapping_dict)

    logging.info("Split dateset into training, validation, and test sets...")
    # Get the split parameters from the dataset parameters
    split_params = dataset_params.get('data_split', {})
    if not split_params:
        logging.ERROR("No split parameters found in dataset parameters. Using default values.")
        exit(1)
    # Split the dataset into training, validation, and test sets

    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset_from_split(
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
