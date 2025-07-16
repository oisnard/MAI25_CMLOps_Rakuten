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
nb_trainable_layers = params['models_parameters']['EfficientNetB1']['nb_trainable_layers']
if not isinstance(nb_trainable_layers, int) or nb_trainable_layers < 0:
    logging.error(f"Invalid nb_trainable_layers value: {nb_trainable_layers}. It should be a non-negative integer.")
    raise ValueError(f"Invalid nb_trainable_layers value: {nb_trainable_layers}. It should be a non-negative integer.")

# Load BATCH_SIZE from params
BATCH_SIZE = params['training_parameters']['batch_size']
if not isinstance(BATCH_SIZE, int) or BATCH_SIZE <= 0:
    logging.error(f"Invalid BATCH_SIZE value: {BATCH_SIZE}. It should be a positive integer.")
    raise ValueError(f"Invalid BATCH_SIZE value: {BATCH_SIZE}. It should be a positive integer.") 


def load_img_model()-> tf.keras.Model:
    """
    Load the pre-trained EfficientNetB1 model for image classification.
    
    Returns:
        tf.keras.Model: The loaded EfficientNetB1 model.
    """


    file_weights = Path(os.path.join(tools.MODEL_DIR, "efficientNetB1_model.weights.h5"))   
    if file_weights.exists():
        try:
            # Load the model architecture and weights
            model = models.build_model_image_efficientNetB1(num_classes=tools.NUM_CLASSES,
                                                            nb_trainable_layers=nb_trainable_layers)
            model.load_weights(os.path.join(tools.MODEL_DIR, "efficientNetB1_model.weights.h5"))
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logging.warning(f"Failed to load the image model: {e}")
            return None
    else:
        logging.warning("Model weights file does not exist. Product classification prediction based on image is not available.")
        return None


IMG_MODEL = load_img_model()


# Load the dictionnary to map integer indices to class labels
dict_mapping_reverse = tools.load_reverse_mapping_dict()
if dict_mapping_reverse is None:
    logging.warning("Failed to load the reverse mapping dictionary.")


def predict_image(image_paths: list) -> list:
    """
    Predict the class of the input images using the pre-trained EfficientNetB1 model.
    
    Args:
        image_paths (list of str): List of file paths to input images.
        
    Returns:
        list: Predicted classes for each input image.
    """
    global IMG_MODEL 
    global dict_mapping_reverse 

    if not isinstance(image_paths, list) or not all(isinstance(path, str) for path in image_paths):
        logger.error("Input must be a list of strings representing image file paths.")
        raise ValueError("Input must be a list of strings representing image file paths.")

    if IMG_MODEL is None:
        IMG_MODEL = load_img_model()
        if IMG_MODEL is None:
            logger.warning("Image model is not loaded. Cannot make predictions.")
            return [-1] * len(image_paths)

    if dict_mapping_reverse is None:
        dict_mapping_reverse = tools.load_reverse_mapping_dict()
        if dict_mapping_reverse is None:
            logger.warning("Reverse mapping dictionary is not loaded. Cannot convert predictions.")
            return [-1] * len(image_paths)

    # Create a TensorFlow dataset from the image paths
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Make predictions
    print("IMG_MODEL:", IMG_MODEL)
    predictions = IMG_MODEL.predict(dataset)
    
    # Convert predictions to class labels
    y_pred = np.argmax(predictions, -1)
    logger.info("Predicted classes obtained successfully.")
    # Convert integer labels back to prdtypecode using the reverse mapping dictionary
    y_pred_prdtypecode = [dict_mapping_reverse.get(pred, "Unknown") for pred in y_pred]

    return y_pred_prdtypecode


if __name__ == "__main__":
    # Example usage
    from  pathlib import Path
    # Assuming the images are in the 'data/raw/image_train' directory
    data_dir = Path(tools.DATA_RAW_IMAGES_TRAIN_DIR)
    files = [f.name for f in data_dir.iterdir() if f.is_file()]
    two_examples = list([os.path.join(tools.DATA_RAW_IMAGES_TRAIN_DIR, filename) for filename in files[:2]])
    logging.info(f"Predicting classes for images: {two_examples}")
    try:
        logger.info("Starting image prediction...")
        predictions = predict_image(two_examples)
        logger.info("Image prediction completed successfully.")
        logger.info("Predicted classes: %s", predictions)
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
