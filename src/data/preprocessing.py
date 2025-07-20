import pandas as pd 
import os 
import src.tools.tools as tools 
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
from sklearn.model_selection import train_test_split
import logging
import warnings


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
    mapping_dict = {int(original) : int(target) for original, target in zip(list_prdtypecode, target_prdtypecode)}
    
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
        stratify=y_temp.prdtypecode
    )

    return X_train_final, X_val, X_test, y_train_final, y_val, y_test
