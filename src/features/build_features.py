
import tensorflow as tf
import pandas as pd 
import src.tools.tools as tools
from src.models.models import load_encode_text_function
import numpy as np
from PIL import Image
import cv2
import logging 
import os

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
    encoded_ds = text_ds.map(encode_text_function, num_parallel_calls=tf.data.AUTOTUNE)
    # Batch and prefetch the dataset for performance
    encoded_ds = encoded_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    attention_masks_len = []
    # Iterate through the encoded dataset to collect attention mask lengths
    for batch in encoded_ds:
        inputs, lbls = batch
        attention_masks_len.extend(inputs['attention_mask'].numpy().sum(axis=1).tolist())

    # Add a DataFrame with the attention mask lengths
    X['nb_attention_mask'] = attention_masks_len

    X['text_length'] = X['feature'].apply(len)
    X['RMS_contrast'] = X['image_path'].apply(compute_RMS_contrast)
    X['sharpness'] = X['image_path'].apply(compute_sharpness)
    X['normalized_useful_surface'] = X['image_path'].apply(compute_normalized_useful_surface)
    X['prdtypecode'] = y['prdtypecode'].values

    return X



def main():
    """
    Main function to load the dataset, build features, and save the processed DataFrame.
    """
    logging.basicConfig(level=logging.INFO)

    # Load parameters
    params = tools.load_dataset_params_from_yaml()

    MODEL_NAME = params['models_parameters']['Camembert']['model_name']
    if not isinstance(MODEL_NAME, str) or not MODEL_NAME:
        logging.error(f"Invalid MODEL_NAME value: {MODEL_NAME}. It should be a non-empty string.")
        raise ValueError(f"Invalid MODEL_NAME value: {MODEL_NAME}. It should be a non-empty string.")
    
    # Load the dataset
    X_train, X_val, y_train, y_val = tools.load_datasets()
    try:
        logging.info("Building features for the dataset...")        
        # Build features
        df = build_features(X_train, y_train, MODEL_NAME)
        print(df.head())
        # Save the processed DataFrame
        df.to_csv(os.path.join(tools.DATA_PROCESSED_DIR, "processed_train_dataset.csv"), index=True)
        
        logging.info("Features built and saved successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()