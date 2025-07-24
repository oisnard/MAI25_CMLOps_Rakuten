
import tensorflow as tf
import pandas as pd 
import src.tools.tools as tools
from src.models.models import load_encode_text_function
import numpy as np
from PIL import Image
import cv2
import logging 
import os
import shutil

def compute_RMS_contrast(filepath: str) -> float:
    """
    Function to compute the RMS contrast of an image.
    Args:
        filepath (str): Path to the image file.
    Returns:
        float: RMS contrast of the image.
    """
    try:
        image = Image.open(filepath).convert("L")  # Convert to grayscale
    except Exception as e:
        logging.error(f"Error opening image {filepath}: {e}")
        raise ValueError(f"Could not open image at {filepath}. Please check the file path and format.")
    image_array = np.array(image)
    rms_contrast = np.sqrt(np.mean(np.square(image_array - np.mean(image_array))))
    return rms_contrast


def compute_sharpness(filepath: str) -> float:
    """
    Compute the sharpness of an image using the variance of the Laplacian.
    
    Args:
        filepath (str): Path to the image file.
        
    Returns:
        float: Sharpness score (higher means sharper).
    """
    try:
        image = Image.open(filepath).convert("L")  # Convert to grayscale
        image_np = np.array(image)
        laplacian = cv2.Laplacian(image_np, cv2.CV_64F)
        sharpness = laplacian.var()
        return sharpness
    except Exception as e:
        logging.error(f"Error computing sharpness for image {filepath}: {e}")
        return -1.0  # Sentinel value if error



def compute_normalized_useful_surface(filepath: str) -> float:
    """
    Compute the normalized useful surface of an image.
    Uses Otsu's method to automatically determine the threshold.

    Args:
        filepath (str): Path to the image file.
        
    Returns:
        float: Normalized useful surface area (0 to 1).
    """
    try:
        # Load the image in RGB format
        image_rgb = Image.open(filepath).convert("RGB")
    except Exception as e:
        logging.error(f"Error opening image {filepath}: {e}")
        raise ValueError(f"Could not open image at {filepath}. Please check the file path and format.")

    # Convert the image to a NumPy array and then to grayscale
    image_np = np.array(image_rgb)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Calcul du seuil optimal avec Otsu
    _, threshold = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Masque des pixels utiles (plus sombres que le seuil)
    useful_mask = image_gray < threshold

    # Surface utile normalisée
    surface_useful = np.sum(useful_mask) / useful_mask.size

    return surface_useful


def build_features(X: pd.DataFrame, 
                   y : pd.DataFrame,
                   model_name: str) -> pd.DataFrame:
    """
    Function to build features for the dataset.
    
    Args:
        X (pd.DataFrame): DataFrame containing the dataset.
        y (pd.DataFrame): DataFrame containing the labels.
        model_name (str): Name of the model to use for encoding.

    Returns:
        pd.DataFrame: DataFrame with the built features.
    """
    if "feature" not in X.columns:
        raise ValueError(f"Column 'feature' not found in DataFrame.")

    if "image_path" not in X.columns:
        raise ValueError(f"Column 'image_path' not found in DataFrame.")

    # Load the encoding function
    default_max_length = 512
    encode_text_function = load_encode_text_function(model_name, default_max_length)
    
    # Create a TensorFlow dataset from the text and labels
    text_ds = tf.data.Dataset.from_tensor_slices((X["feature"].values.astype(str),
                                                  y["prdtypecode"].values.astype(np.int32)))

    # Map the encoding function to the dataset
    # Use num_parallel_calls to speed up the mapping process
    logging.info("Encoding text data...")
    encoded_ds = text_ds.map(encode_text_function, num_parallel_calls=tf.data.AUTOTUNE)
    # Batch and prefetch the dataset for performance
    encoded_ds = encoded_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    attention_masks_len = []
    # Iterate through the encoded dataset to collect attention mask lengths
    logging.info("Collecting attention mask lengths...")
    for batch in encoded_ds:
        inputs, lbls = batch
        attention_masks_len.extend(inputs['attention_mask'].numpy().sum(axis=1).tolist())

    # Add a DataFrame with the attention mask lengths
    X['nb_attention_mask'] = attention_masks_len
    logging.info("Designation + description length added to DataFrame.")
    X['text_length'] = X['feature'].apply(len)
    logging.info("Computation of RMS contrast...")
    X['RMS_contrast'] = X['image_path'].apply(compute_RMS_contrast)
    logging.info("Computation of sharpness...")
    X['sharpness'] = X['image_path'].apply(compute_sharpness)
    logging.info("Computation of normalized useful surface...")
    X['normalized_useful_surface'] = X['image_path'].apply(compute_normalized_useful_surface)

    dict_mapping_reverse = tools.load_reverse_mapping_dict()
    if dict_mapping_reverse is None:
        logging.error("Failed to load the reverse mapping dictionary.")
        raise ValueError("Failed to load the reverse mapping dictionary.")
    # Convert integer predictions to prdtypecode using the reverse mapping dictionary
    y_prdtypecode = [dict_mapping_reverse.get(pred, "Unknown") for pred in y['prdtypecode'].values.tolist()]

    X['prdtypecode'] = y_prdtypecode

    X = X.drop(columns=['feature', 'image_path'], errors='ignore')

    return X

def current_datastream_name() -> str:
    """
    Get the name of the current datastream.
    
    Returns:
        str: The name of the current datastream.
    """
    if not os.path.exists(tools.DATA_PROCESSED_STREAM_DIR):
        logging.error("DATA_PROCESSED_STREAM_DIR does not exist.")
        raise FileNotFoundError("DATA_PROCESSED_STREAM_DIR does not exist.")

    files = os.listdir(tools.DATA_PROCESSED_STREAM_DIR)
    if not files:
        logging.error("No datastream files found in DATA_PROCESSED_STREAM_DIR. Datastream creation might not have been completed.")
        raise FileNotFoundError("No datastream files found in DATA_PROCESSED_STREAM_DIR.")

    X_files = sorted(f for f in os.listdir(tools.DATA_PROCESSED_STREAM_DIR) if f.startswith("X_") and f.endswith(".csv"))
    if not X_files:
        logging.error("No X datastream files found in DATA_PROCESSED_STREAM_DIR.")
        raise FileNotFoundError("No X datastream files found in DATA_PROCESSED_STREAM_DIR.")

    stream_name = X_files[-1].split('_', 1)[1].rsplit('.', 1)[0]  # Extract the stream name from the last X file

    # Assuming the first file is the current datastream
    return stream_name

def check_reset_need() -> bool:
    """
    Check if a reset is needed based on the content of folders DATA_RAW_STREAM_DIR and DATA_MONITORING_SAMPLE_DIR.
    
    Returns:
        bool: True if a reset is needed, False otherwise.
    """
    if not os.path.exists(tools.DATA_RAW_STREAM_DIR):
        if os.path.exists(tools.DATA_MONITORING_SAMPLE_DIR):
            return True
        return False
    else:
        if os.path.exists(tools.DATA_PROCESSED_STREAM_DIR):
            X_files = sorted(f for f in os.listdir(tools.DATA_PROCESSED_STREAM_DIR) if f.startswith("X_") and f.endswith(".csv"))
            if not X_files:
                return True
            if len(X_files) <= 1:
                return True
            return False



def main():
    """
    Main function to load the dataset, build features, and save the processed DataFrame.
    """
    logging.basicConfig(level=logging.INFO)

    if check_reset_need():
        logging.info("Reset needed. Removing existing monitoring sample data.")
        if os.path.exists(tools.DATA_MONITORING_SAMPLE_DIR):
            shutil.rmtree(tools.DATA_MONITORING_SAMPLE_DIR)

    # Load parameters
    params = tools.load_dataset_params_from_yaml()

    MODEL_NAME = params['models_parameters']['Camembert']['model_name']
    if not isinstance(MODEL_NAME, str) or not MODEL_NAME:
        logging.error(f"Invalid MODEL_NAME value: {MODEL_NAME}. It should be a non-empty string.")
        raise ValueError(f"Invalid MODEL_NAME value: {MODEL_NAME}. It should be a non-empty string.")
    
    logging.info(f"Using model: {MODEL_NAME}")
    datastream_name = current_datastream_name()
    logging.info(f"Current datastream name: {datastream_name}")

    # Load the dataset
    X_train, X_val, y_train, y_val, X_test, y_test = tools.load_full_datasets()
    try:
        logging.info("Building features for the dataset...")        
        # Build features
        X = pd.concat([X_train, X_val, X_test], axis=0)
        y = pd.concat([y_train, y_val, y_test], axis=0)
        df = build_features(X, y, MODEL_NAME)
        if not os.path.exists(tools.DATA_MONITORING_SAMPLE_DIR):
            os.makedirs(tools.DATA_MONITORING_SAMPLE_DIR, exist_ok=True)
        if "reference" in datastream_name:
            filename_sample = os.path.join(tools.DATA_MONITORING_SAMPLE_DIR, "reference_dataset.csv")
        else:
            filename_sample = os.path.join(tools.DATA_MONITORING_SAMPLE_DIR, f"{datastream_name}_dataset.csv")
        print(tools.DATA_MONITORING_SAMPLE_DIR)
        print(filename_sample)
        # Save the processed DataFrame
        df.to_csv(filename_sample, index=True)

        logging.info(f"Features built and saved successfully in file {filename_sample}.")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()