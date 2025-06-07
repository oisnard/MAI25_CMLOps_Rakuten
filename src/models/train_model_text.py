import tensorflow as tf
import pandas as pd
import src.tools.tools as tools 
import src.models.models as models
from src.models.metrics import SparseF1Score
import os 
import logging 





def load_datasets() -> tuple:
    """
    Function to load the datasets for training and validation.
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test : DataFrames containing the training and validation data.
    """
    try:
        X_train = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "X_train.csv"), index_col=0)
        y_train = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "y_train.csv"), index_col=0)
        X_val = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "X_val.csv"), index_col=0)
        y_val = pd.read_csv(os.path.join(tools.DATA_PROCESSED_DIR, "y_val.csv"), index_col=0)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading datasets: {e}")
        raise
    # Ensure that the datasets are not empty
    if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
        logging.error("One or more datasets are empty.")
        raise ValueError("One or more datasets are empty.")
    # Ensure that the indices of X_train and y_train match
    if not X_train.index.equals(y_train.index):
        logging.error("Indices of X_train and y_train do not match.")
        raise ValueError("Indices of X_train and y_train do not match.")
    if not X_val.index.equals(y_val.index):
        logging.error("Indices of X_val and y_val do not match.")
        raise ValueError("Indices of X_val and y_val do not match.")
    # Ensure that the 'feature' column exists in X_train, X_val, and X_test
    if 'feature' not in X_train.columns or 'feature' not in X_val.columns:
        logging.error("The 'feature' column is missing in one of the datasets.")
        raise ValueError("The 'feature' column is missing in one of the datasets.")
    # Ensure that the 'prdtypecode' column exists in y_train, y_val, and y_test
    if 'prdtypecode' not in y_train.columns or 'prdtypecode' not in y_val.columns:
        logging.error("The 'prdtypecode' column is missing in one of the label datasets.")
        raise ValueError("The 'prdtypecode' column is missing in one of the label datasets.")
    # Ensure that the 'prdtypecode' column is of integer type
    if not pd.api.types.is_integer_dtype(y_train['prdtypecode']) or not pd.api.types.is_integer_dtype(y_val['prdtypecode']):
        logging.error("The 'prdtypecode' column must be of integer type.")
        raise ValueError("The 'prdtypecode' column must be of integer type.")
    # Ensure that the 'feature' column is of string type
    if not pd.api.types.is_string_dtype(X_train['feature']) or not pd.api.types.is_string_dtype(X_val['feature']):
        logging.error("The 'feature' column must be of string type.")
        raise ValueError("The 'feature' column must be of string type.")
    # Ensure that the 'prdtypecode' column is not empty
    if y_train['prdtypecode'].isnull().any() or y_val['prdtypecode'].isnull().any():
        logging.error("The 'prdtypecode' column contains null values.")
        raise ValueError("The 'prdtypecode' column contains null values.")
    # Ensure that the 'feature' column is not empty
    if X_train['feature'].isnull().any() or X_val['feature'].isnull().any():
        logging.error("The 'feature' column contains null values.")
        raise ValueError("The 'feature' column contains null values.")
    return X_train, X_val, y_train, y_val

def main():
    # Set logging level
    logging.basicConfig(level=logging.INFO)

    # Load dataset parameters from YAML file
    params = tools.load_dataset_params_from_yaml()

    # Load the MAX LEN and MODEL_NAME from params
    MAX_LEN = params['models_parameters']['Camembert']['max_length']
    if not isinstance(MAX_LEN, int) or MAX_LEN <= 0:
        logging.error(f"Invalid max_len value: {MAX_LEN}. It should be a positive integer.")
        raise ValueError(f"Invalid max_len value: {MAX_LEN}. It should be a positive integer.")
    MODEL_NAME = params['models_parameters']['Camembert']['model_name']
    if not isinstance(MODEL_NAME, str) or not MODEL_NAME:   
        logging.error(f"Invalid MODEL_NAME value: {MODEL_NAME}. It should be a non-empty string.")
        raise ValueError(f"Invalid MODEL_NAME value: {MODEL_NAME}. It should be a non-empty string.")

    # Load BATCH_SIZE from params
    BATCH_SIZE = params['training_parameters']['batch_size']
    if not isinstance(BATCH_SIZE, int) or BATCH_SIZE <= 0:
        logging.error(f"Invalid BATCH_SIZE value: {BATCH_SIZE}. It should be a positive integer.")
        raise ValueError(f"Invalid BATCH_SIZE value: {BATCH_SIZE}. It should be a positive integer.")

    # Load the nb of epochs from params
    EPOCHS = params['training_parameters']['epochs']
    if not isinstance(EPOCHS, int) or EPOCHS <= 0:
        logging.error(f"Invalid EPOCHS value: {EPOCHS}. It should be a positive integer.")
        raise ValueError(f"Invalid EPOCHS value: {EPOCHS}. It should be a positive integer.")


    # Load datasets
    logging.info("Loading datasets...")
    # Load the datasets for training, validation, and testing
    X_train, X_val, y_train, y_val = load_datasets()    
    logging.info(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")


    # Preprocess text data
    logging.info("Preprocessing text data...")
    # Tokenize the text data using the tokenizer
    # Define batch size
    if not isinstance(BATCH_SIZE, int) or BATCH_SIZE <= 0:
        logging.error(f"Invalid BATCH_SIZE value: {BATCH_SIZE}. It should be a positive integer.")
        raise ValueError(f"Invalid BATCH_SIZE value: {BATCH_SIZE}. It should be a positive integer.")

    # Setting the encode function based on the MODEL_NAME and MAX_LEN 
    encode = models.load_encode_text_function(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
    # Convert the encoded data and labels to TensorFlow datasets
    logging.info("Creating TensorFlow datasets for training and validation...")
    # Convert the training and validation datasets to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.feature.tolist(), 
                                                        y_train.prdtypecode.tolist())
                                                        ).map(encode).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val.feature.tolist(),
                                                        y_val.prdtypecode.tolist())
                                                        ).map(encode).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # Construction du modèle
    model = models.build_cam_model(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN, NUM_CLASSES=tools.NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", SparseF1Score(num_classes=tools.NUM_CLASSES, average='weighted')]  
    )


    history = model.fit(
        train_dataset,  # Pass inputs as a dictionary
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data = val_dataset)
    
    logging.info("Training completed.")
    # Save the model
    model.save_weights(os.path.join(tools.MODEL_DIR, "camembert_model.weights.h5"))
    logging.info("Model weights saved successfully.")
    # Save the model architecture
    model_json = model.to_json()
    with open(os.path.join(tools.MODEL_DIR, "camembert_model.json"), "w") as json_file:
        json_file.write(model_json)
    logging.info("Model architecture saved successfully.")
   

if __name__ == "__main__":
    main()
else:
    import inspect

    # If this script is imported, run the main function only if it is called from train_model.py
    stack = inspect.stack()
    for frame in stack:
        if "train_model.py" in frame.filename:
            main()
            break
