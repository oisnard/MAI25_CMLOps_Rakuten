import tensorflow as tf
import numpy as np
import src.models.models as models 
import src.tools.tools as tools
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset parameters from YAML file
params = tools.load_dataset_params_from_yaml()
MODEL_NAME = params['models_parameters']['Camembert']['model_name']
MAX_LEN = params['models_parameters']['Camembert']['max_length']
BATCH_SIZE = params['training_parameters']['batch_size']
if not isinstance(MAX_LEN, int) or MAX_LEN <= 0:
    logger.error(f"Invalid max_len value: {MAX_LEN}. It should be a positive integer.")
    raise ValueError(f"Invalid max_len value: {MAX_LEN}. It should be a positive integer.")# Load the encoder function for the Camembert model
try:    
    encode = models.load_encode_text_function(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
except Exception as e:
    logger.error(f"An error occurred while loading the encode function: {e}")
    raise 

# Build the text model
try:
    text_model = models.build_cam_model(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
    text_model.load_weights(os.path.join(tools.MODEL_DIR, "camembert_model.weights.h5"))
except Exception as e:
    logger.error(f"An error occurred while building or loading the model: {e}")
    raise

# Load the dictionnary to map integer indices to class labels
dict_mapping_reverse = tools.load_reverse_mapping_dict()
if dict_mapping_reverse is None:
    logging.error("Failed to load the reverse mapping dictionary.")
    raise ValueError("Failed to load the reverse mapping dictionary.")




def predict_text(texts: list) -> list:
    """
    Predict the class of the input texts using the pre-trained Camembert model.
    
    Args:
        texts (list of str): List of input texts to predict.
        
    Returns:
        list: Predicted classes for each input text.
    """
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        logger.error("Input must be a list of strings.")
        raise ValueError("Input must be a list of strings.")
    
    # Encode the texts
    dummy_classes = np.zeros(shape=(len(texts), ), dtype=np.int32)  # Dummy classes for encoding
    print(dummy_classes)
    print("Building dataset for prediction...")
    data_dataset = tf.data.Dataset.from_tensor_slices((texts, dummy_classes)
                                                     ).map(encode).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print(data_dataset)
    print("Dataset for prediction built successfully.")
    # Make predictions
    predictions = text_model.predict(data_dataset)
    if predictions is None or len(predictions) == 0:
        logger.error("Model predictions are empty.")
        raise ValueError("Model predictions are empty.")
    # Convert predictions to class labels
    y_pred = np.argmax(predictions, -1)
    
    # Convert integer predictions to prdtypecode using the reverse mapping dictionary
    y_pred_prdtypecode = [dict_mapping_reverse.get(pred, "Unknown") for pred in y_pred]

    return y_pred_prdtypecode


if __name__ == "__main__":
    # Example usage
    logging.info("Starting text prediction...")
    # Example texts for prediction
    example_texts = ["This is a sample text for prediction.", "Another example text."]
    logging.info(f"Example texts: {example_texts}")
    try:
        predictions = predict_text(example_texts)
        logger.info(f"Predictions: {predictions}")
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise