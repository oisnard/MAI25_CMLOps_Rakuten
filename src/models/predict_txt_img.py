import tensorflow as tf
import numpy as np
import src.models.models as models 
import src.tools.tools as tools
import logging
import os
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset parameters from YAML file
params = tools.load_dataset_params_from_yaml()
MODEL_NAME = params['models_parameters']['Camembert']['model_name']
MAX_LEN = params['models_parameters']['Camembert']['max_length']
BATCH_SIZE = params['training_parameters']['batch_size']
if not isinstance(MAX_LEN, int) or MAX_LEN <= 0:
    logger.error(f"Invalid max_len value: {MAX_LEN}. It should be a positive integer.")
    raise ValueError(f"Invalid max_len value: {MAX_LEN}. It should be a positive integer.") 

if not isinstance(MAX_LEN, int) or MAX_LEN <= 0:
    logger.error(f"Invalid max_len value: {MAX_LEN}. It should be a positive integer.")
    raise ValueError(f"Invalid max_len value: {MAX_LEN}. It should be a positive integer.")# Load the encoder function for the Camembert model
try:    
    ENCODE_DATA = models.load_encode_data(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
except Exception as e:
    logger.error(f"An error occurred while loading the encode function: {e}")
    raise ValueError(f"An error occurred while loading the encode function: {e}")


def load_merged_model():
    """
    Load the pre-trained Merged model for Product Classification based on text and images.
    
    Returns:
        tf.keras.Model: The loaded Merged model.
    """
    # Check if the model weights file exists
    global MODEL_NAME, MAX_LEN

    file_weights = Path(os.path.join(tools.MODEL_DIR, "camembert_model.weights.h5"))
    if not file_weights.exists():
        logger.warning("Model weights file does not exist. Product classification prediction based on text is not available.")
        return None
    # Build the text model
    try:
        text_model = models.build_merged_model(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN, NUM_CLASSES=tools.NUM_CLASSES)
        text_model.load_weights(os.path.join(tools.MODEL_DIR, "merged_model.weights.h5"))
        return text_model
    except Exception as e:
        logger.warning(f"An error occurred while building or loading the merged model: {e}")
        return None
    
# Load the dictionnary to map integer indices to class labels
dict_mapping_reverse = tools.load_reverse_mapping_dict()
if dict_mapping_reverse is None:
    logging.warning("Failed to load the reverse mapping dictionary.")

MERGED_MODEL = load_merged_model()

def predict_text_image(texts: list, image_filepath: list) -> list:
    """
    Predict the class of the input texts and images using the pre-trained Merged model.
    
    Args:
        texts (list of str): List of input texts to predict.
        image_filepath (list of str): List of file paths to input images.

    Returns:
        list: Predicted classes for each input text and image.
    """
    global MERGED_MODEL
    global dict_mapping_reverse
    global ENCODE_DATA

    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        logger.error("Input must be a list of strings representing text.")
        raise ValueError("Input must be a list of strings representing text.")

    if not isinstance(image_filepath, list) or not all(isinstance(path, str) for path in image_filepath):
        logger.error("Input must be a list of strings representing image file paths.")
        raise ValueError("Input must be a list of strings representing image file paths.")

    if len(texts) != len(image_filepath):
        logger.error("The length of texts and image_filepath lists must be the same.")
        raise ValueError("The length of texts and image_filepath lists must be the same.")

    if MERGED_MODEL is None:
        MERGED_MODEL = load_merged_model()
        if MERGED_MODEL is None:
            logger.warning("Merged model is not loaded. Cannot make predictions.")
            return [-1] * len(texts)

    if dict_mapping_reverse is None:
        dict_mapping_reverse = tools.load_reverse_mapping_dict()
        if dict_mapping_reverse is None:
            logger.warning("Reverse mapping dictionary is not loaded. Cannot convert predictions.")
            return [-1] * len(texts)

    # Create a TensorFlow dataset from the text and image inputs
    dummy_classes = [0] * len(texts)  # Dummy classes for encoding
    data_dataset = tf.data.Dataset.from_tensor_slices((texts, image_filepath, dummy_classes)
                                                     ).map(ENCODE_DATA).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Make predictions
    predictions = MERGED_MODEL.predict(data_dataset)

    # Convert predictions to class labels
    y_pred = np.argmax(predictions, -1)
    logger.info("Predicted classes obtained successfully.")

    # Convert integer labels back to prdtypecode using the reverse mapping dictionary
    y_pred_prdtypecode = [dict_mapping_reverse.get(pred, "Unknown") for pred in y_pred]

    return y_pred_prdtypecode


if __name__ == "__main__":
    # Example usage
    logging.info("Starting text and image prediction...")
    X_train, X_val, y_train, y_val = tools.load_datasets()

    df = X_train.sample(n=5)
    example_texts = df['feature'].tolist()
    example_images = df['image_path'].tolist()
    logging.info(f"Example texts: {example_texts}")
    logging.info(f"Example images: {example_images}")

    try:
        predictions = predict_text_image(example_texts, example_images)
        logging.info(f"Predictions: {predictions}")
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        raise e
