import os
import pandas as pd
import pytest
import src.tools.tools as tools
from src.features.build_features import build_features 

@pytest.fixture(scope="module")
def MODEL_NAME():
    params = tools.load_dataset_params_from_yaml()

    MODEL_NAME = params['models_parameters']['Camembert']['model_name']
    if not isinstance(MODEL_NAME, str) or not MODEL_NAME:
        raise ValueError(f"Invalid MODEL_NAME value: {MODEL_NAME}. It should be a non-empty string.")
    return MODEL_NAME


@pytest.fixture(scope="module")
def X_val_df():
    """
    Load only once the X_val processed data file.
    """
    df = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "X_val.csv"), index_col=0)
    return df


@pytest.fixture(scope="module")
def y_val_df():
    """
    Load only once the y_val processed data file.
    """
    df = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "y_val.csv"), index_col=0)
    return df


def test_feature_building(X_val_df, y_val_df, MODEL_NAME):
    """
    Build features for the validation dataset.
    """
    df = build_features(X_val_df, y_val_df, MODEL_NAME)
    # Check if the DataFrame has the expected columns
    expected_columns = ["nb_attention_mask", "text_length", "RMS_contrast", "sharpness", "normalized_useful_surface"]
    for col in expected_columns:
        if col not in df.columns:
            assert False, f"Missing expected column: {col}"
    # Check if the DataFrame is not empty
    if df.empty:
        assert False, "DataFrame is empty."
    if df.shape[0] != X_val_df.shape[0]:
        assert False, "The number of rows in the processed DataFrame does not match the input DataFrame."