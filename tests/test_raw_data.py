import os
import pandas as pd
import pytest
import src.tools.tools as tools


@pytest.fixture(scope="module")
def xtrain_df():
    """load only once the X_train raw data file."""
    # Load the X_train raw data file
    return tools.load_xtrain_raw_data()

@pytest.fixture(scope="module")
def xtest_df():
    """load only once the X_test raw data file."""
    # Load the X_test raw data file
    return tools.load_xtest_raw_data()

# test if the X_train raw data file is a valid pandas DataFrame 
def test_xtrain_raw_dataframe(xtrain_df):
    """
    Test if the X_train raw data file is a valid pandas DataFrame.
    """
    X_train = xtrain_df
    
    # Check if X_train is a DataFrame
    assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame"
    
    # Check if X_train is not empty
    assert not X_train.empty, "X_train should not be empty"
    
    # Check if required columns are present
    assert 'productid' in X_train.columns, "X_train should contain 'product_id' column"
    assert 'imageid' in X_train.columns, "X_train should contain 'image_id' column"


# test if the images referred in X_train raw data are present in the raw data directory
def test_xtrain_raw_images_presence(xtrain_df):
    """
    Test if images referred in X_train raw data are present in the raw data directory.
    """
    X_train = xtrain_df

    nb_missing_files = 0
    # Check if the images exist in the raw data directory
    for product_id, image_id in zip(X_train['productid'], X_train['imageid']):
        filename = f"image_{image_id}_product_{product_id}.jpg"
        filepath = os.path.join(tools.DATA_RAW_IMAGES_TRAIN_DIR, filename)
        if not os.path.exists(filepath):
            nb_missing_files += 1
            print(f"Missing file: {filepath}")
    assert nb_missing_files == 0, f"Found {nb_missing_files} missing files in the raw data directory."


# test if the X_test raw data file is a valid pandas DataFrame 
def test_xtest_raw_dataframe(xtest_df):
    """
    Test if the X_test raw data file is a valid pandas DataFrame.
    """
    X_test = xtest_df
    
    # Check if X_train is a DataFrame
    assert isinstance(X_test, pd.DataFrame), "X_train should be a pandas DataFrame"
    
    # Check if X_train is not empty
    assert not X_test.empty, "X_train should not be empty"
    
    # Check if required columns are present
    assert 'productid' in X_test.columns, "X_test should contain 'product_id' column"
    assert 'imageid' in X_test.columns, "X_test should contain 'image_id' column"


# test if the images referred in X_test raw data are present in the raw data directory
def test_xtest_raw_images_presence(xtest_df):
    """
    Test if images referred in X_test raw data are present in the raw data directory.
    """
    X_test = xtest_df

    nb_missing_files = 0
    # Check if the images exist in the raw data directory
    for product_id, image_id in zip(X_test['productid'], X_test['imageid']):
        filename = f"image_{image_id}_product_{product_id}.jpg"
        filepath = os.path.join(tools.DATA_RAW_IMAGES_TEST_DIR, filename)
        if not os.path.exists(filepath):
            nb_missing_files += 1
            print(f"Missing file: {filepath}")
    assert nb_missing_files == 0, f"Found {nb_missing_files} missing files in the raw data directory."


# test if the Y_train raw data file is a valid pandas DataFrame
def test_ytrain_raw_dataframe():
    """
    Test if the Y_train raw data file is a valid pandas DataFrame.
    """
    Y_train = tools.load_ytrain_raw_data()
    
    # Check if Y_train is a DataFrame
    assert isinstance(Y_train, pd.DataFrame), "Y_train should be a pandas DataFrame"
    
    # Check if Y_train is not empty
    assert not Y_train.empty, "Y_train should not be empty"
    
    # Check if required columns are present
    assert 'prdtypecode' in Y_train.columns, "Y_train should contain 'prdtypecode' column"
