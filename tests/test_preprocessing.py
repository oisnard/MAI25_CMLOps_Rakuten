import os
import pandas as pd
import pytest
import src.tools.tools as tools

@pytest.fixture(scope="module")
def xtrain_df():
    """
    load only once the X_train processed data file.
    """
    df = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "X_train.csv"), index_col=0)
    return df 


@pytest.fixture(scope="module")
def ytrain_df():
    """
    load only once the y_train processed data file.
    """
    df = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "y_train.csv"), index_col=0)
    return df 


@pytest.fixture(scope="module")
def xval_df():
    """
    load only once the X_val processed data file.
    """
    df = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "X_val.csv"), index_col=0)
    return df 

@pytest.fixture(scope="module")
def yval_df():
    """
    load only once the y_train processed data file.
    """
    df = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "y_val.csv"), index_col=0)
    return df 


@pytest.fixture(scope="module")
def xtest_df():
    """
    load only once the X_val processed data file.
    """
    df = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "X_test.csv"), index_col=0)
    return df 

@pytest.fixture(scope="module")
def ytest_df():
    """
    load only once the y_train processed data file.
    """
    df = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "y_test.csv"), index_col=0)
    return df 


# test if the X_train processed data file is a valid pandas DataFrame 
def test_xtrain_processed_dataframe_is_valid(xtrain_df):
    """
    Test if the X_train processed data file is a valid pandas DataFrame.
    """
    X_train = xtrain_df
    # Check if X_train is a DataFrame
    assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame"
    
    # Check if X_train is not empty
    assert not X_train.empty, "X_train should not be empty"
    
    # Check if required columns are present
    assert 'feature' in X_train.columns, "X_train should contain 'feature' column"

    assert X_train.isna().sum().sum() == 0, "X_train should not contain any NaN"


# test if the y_train processed data file is a valid pandas DataFrame 
def test_ytrain_processed_dataframe_is_valid(ytrain_df):
    """
    Test if the y_train processed data file is a valid pandas DataFrame.
    """
    y_train = ytrain_df
    # Check if X_train is a DataFrame
    assert isinstance(y_train, pd.DataFrame), "y_train should be a pandas DataFrame"
    
    # Check if X_train is not empty
    assert not y_train.empty, "y_train should not be empty"
    
    # Check if required columns are present
    assert 'prdtypecode' in y_train.columns, "y_train should contain 'prdtypecode' column"

    assert y_train.isna().sum().sum() == 0, "y_train should not contain any NaN"



# test if the X_val processed data file is a valid pandas DataFrame 
def test_xval_processed_dataframe_is_valid(xval_df):
    """
    Test if the X_val processed data file is a valid pandas DataFrame.
    """
    X_val = xval_df
    # Check if X_val is a DataFrame
    assert isinstance(X_val, pd.DataFrame), "X_val should be a pandas DataFrame"
    
    # Check if X_val is not empty
    assert not X_val.empty, "X_val should not be empty"
    
    # Check if required columns are present
    assert 'feature' in X_val.columns, "X_val should contain 'feature' column"

    assert X_val.isna().sum().sum() == 0, "X_val should not contain any NaN"


# test if the y_val processed data file is a valid pandas DataFrame 
def test_yval_processed_dataframe_is_valid(yval_df):
    """
    Test if the y_val processed data file is a valid pandas DataFrame.
    """
    y_val = yval_df
    # Check if X_train is a DataFrame
    assert isinstance(y_val, pd.DataFrame), "y_val should be a pandas DataFrame"
    
    # Check if X_train is not empty
    assert not y_val.empty, "y_val should not be empty"
    
    # Check if required columns are present
    assert 'prdtypecode' in y_val.columns, "y_val should contain 'product_id' column"

    assert y_val.isna().sum().sum() == 0, "y_val should not contain any NaN"


# test if the X_test processed data file is a valid pandas DataFrame 
def test_xtest_processed_dataframe_is_valid(xtest_df):
    """
    Test if the X_test processed data file is a valid pandas DataFrame.
    """
    X_test = xtest_df
    # Check if X_val is a DataFrame
    assert isinstance(X_test, pd.DataFrame), "X_test should be a pandas DataFrame"
    
    # Check if X_val is not empty
    assert not X_test.empty, "X_test should not be empty"
    
    # Check if required columns are present
    assert 'feature' in X_test.columns, "X_test should contain 'feature' column"

    assert X_test.isna().sum().sum() == 0, "X_test should not contain any NaN"
    

# test if the y_test processed data file is a valid pandas DataFrame 
def test_ytest_processed_dataframe_is_valid(ytest_df):
    """
    Test if the y_test processed data file is a valid pandas DataFrame.
    """
    y_test = ytest_df
    # Check if X_train is a DataFrame
    assert isinstance(y_test, pd.DataFrame), "y_test should be a pandas DataFrame"
    
    # Check if X_train is not empty
    assert not y_test.empty, "y_test should not be empty"
    
    # Check if required columns are present
    assert 'prdtypecode' in y_test.columns, "y_test should contain 'product_id' column"

    assert y_test.isna().sum().sum() == 0, "y_test should not contain any NaN"


# Test if X_train and y_train have coherent shape
def test_Xtrain_ytrain_coherency(xtrain_df, ytrain_df):
    """
    Test if X_train and y_train have referring to the same nb of products
    """
    assert xtrain_df.shape[0] == ytrain_df.shape[0], "X_train and y_train should have same nb of lines"
    assert xtrain_df.index.equals(ytrain_df.index), "X_train and y_train should have the same index"


# Test if X_val and y_val have coherent shape
def test_Xval_yval_coherency(xval_df, yval_df):
    """
    Test if X_val and y_val have referring to the same nb of products
    """
    assert xval_df.shape[0] == yval_df.shape[0], "X_val and y_val should have same nb of lines"
    assert xval_df.index.equals(yval_df.index), "X_val and y_val should have the same index"

# Test if X_test and y_test have coherent shape
def test_Xtest_ytest_coherency(xtest_df, ytest_df):
    """
    Test if X_test and y_test have referring to the same nb of products
    """
    assert xtest_df.shape[0] == ytest_df.shape[0], "X_test and y_test should have same nb of lines"
    assert xtest_df.index.equals(ytest_df.index), "X_test and y_test should have the same index"


# test if the images referred in X_train processed data are present in the raw data directory
def test_xtrain_images_presence(xtrain_df):
    """
    Test if images referred in X_train processed data are present in the raw data directory.
    """
    X_train = xtrain_df

    nb_missing_files = 0
    # Check if the images exist in the raw data directory
    for image_path in X_train['image_path']:
        if not os.path.exists(image_path):
            nb_missing_files += 1
            print(f"Missing file: {image_path}")
    assert nb_missing_files == 0, f"Found {nb_missing_files} missing files in the raw data directory."


# test if the images referred in X_val processed data are present in the raw data directory
def test_xval_images_presence(xval_df):
    """
    Test if images referred in X_val processed data are present in the raw data directory.
    """
    X_val = xval_df

    nb_missing_files = 0
    # Check if the images exist in the raw data directory
    for image_path in X_val['image_path']:
        if not os.path.exists(image_path):
            nb_missing_files += 1
            print(f"Missing file: {image_path}")
    assert nb_missing_files == 0, f"Found {nb_missing_files} missing files in the raw data directory."


# test if the images referred in X_test processed data are present in the raw data directory
def test_xtest_images_presence(xtest_df):
    """
    Test if images referred in X_test processed data are present in the raw data directory.
    """
    X_test = xtest_df

    nb_missing_files = 0
    # Check if the images exist in the raw data directory
    for image_path in X_test['image_path']:
        if not os.path.exists(image_path):
            nb_missing_files += 1
            print(f"Missing file: {image_path}")
    assert nb_missing_files == 0, f"Found {nb_missing_files} missing files in the raw data directory."


