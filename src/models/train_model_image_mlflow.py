import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1 
import os 
import logging 
import json
import src.tools.tools as tools
import src.models.models as models
import src.models.metrics as metrics
import mlflow 

# Function to load datasets
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
    if 'image_path' not in X_train.columns or 'image_path' not in X_val.columns:
        logging.error("The 'image_path' column is missing in one of the datasets.")
        raise ValueError("The 'image_path' column is missing in one of the datasets.")
    # Ensure that the 'prdtypecode' column exists in y_train, y_val, and y_test
    if 'prdtypecode' not in y_train.columns or 'prdtypecode' not in y_val.columns:
        logging.error("The 'prdtypecode' column is missing in one of the label datasets.")
        raise ValueError("The 'prdtypecode' column is missing in one of the label datasets.")
    # Ensure that the 'prdtypecode' column is of integer type
    if not pd.api.types.is_integer_dtype(y_train['prdtypecode']) or not pd.api.types.is_integer_dtype(y_val['prdtypecode']):
        logging.error("The 'prdtypecode' column must be of integer type.")
        raise ValueError("The 'prdtypecode' column must be of integer type.")
    # Ensure that the 'image_path' column is of string type
    if not pd.api.types.is_string_dtype(X_train['image_path']) or not pd.api.types.is_string_dtype(X_val['feature']):
        logging.error("The 'image_path' column must be of string type.")
        raise ValueError("The 'image_path' column must be of string type.")
    # Ensure that the 'prdtypecode' column is not empty
    if y_train['prdtypecode'].isnull().any() or y_val['prdtypecode'].isnull().any():
        logging.error("The 'prdtypecode' column contains null values.")
        raise ValueError("The 'prdtypecode' column contains null values.")
    # Ensure that the 'feature' column is not empty
    if X_train['image_path'].isnull().any() or X_val['image_path'].isnull().any():
        logging.error("The 'image_path' column contains null values.")
        raise ValueError("The 'image_path' column contains null values.")
    return X_train, X_val, y_train, y_val


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Training model based on images...")
    logging.info("Available GPUs: %s", tf.config.list_physical_devices('GPU'))
    # Load parameters from YAML file
    params = tools.load_dataset_params_from_yaml()

   # Load dataset parameters from YAML file
    params = tools.load_dataset_params_from_yaml()    

    # Load the number of classes, the number of trainable layers, batch_size and nb of epochs from params
    num_classes = tools.NUM_CLASSES
    nb_trainable_layers = params['models_parameters']['EfficientNetB1']['nb_trainable_layers']
    if not isinstance(nb_trainable_layers, int) or nb_trainable_layers < 0:
        logging.error(f"Invalid nb_trainable_layers value: {nb_trainable_layers}. It should be a non-negative integer.")
        raise ValueError(f"Invalid nb_trainable_layers value: {nb_trainable_layers}. It should be a non-negative integer.")

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
    X_train, X_val, y_train, y_val = load_datasets()    
    logging.info(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")
    # Check if the datasets are not empty
    if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
        logging.error("One or more datasets are empty.")
        raise ValueError("One or more datasets are empty.")

    logging.info("Creating TensorFlow datasets for training and validation...")
    # Convert the training and validation datasets to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.image_path.tolist(), y_train.prdtypecode.tolist()))
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val.image_path.tolist(), y_val.prdtypecode.tolist()))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # Check if the datasets are not empty
    if train_dataset is None or val_dataset is None:
        logging.error("One or more datasets are empty.")
        raise ValueError("One or more datasets are empty.")


    logging.info("Setting experiment...")
    mlflow.set_experiment("Rakuten_Image_Model")
    
    # Start MLflow run
    logging.info("Starting MLflow run...")
    with mlflow.start_run() as run:
        logging.info(f"MLflow run started. Run ID: {run.info.run_id}")
        mlflow.log_param("data_ratio", params['data_ratio'])
        mlflow.log_param("model_name", params['models_parameters']['EfficientNetB1']['model_name'])
        mlflow.log_param("nb_trainable_layers", nb_trainable_layers)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)

        # Build the model
        logging.info("Building the EfficientNetB1 model...")
        model = models.build_model_image_efficientNetB1(num_classes=num_classes, nb_trainable_layers=nb_trainable_layers)
        logging.info("Model built successfully.")

        weights_file = os.path.join(tools.MODEL_DIR, "efficientNetB1_model.weights.h5")
        if os.path.exists(tools.MODEL_DIR):
            model.load_weights(os.path.join(tools.MODEL_DIR, "efficientNetB1_model.weights.h5"))
            logging.info("Model weights loaded successfully.")


        # Compile the model
        logging.info("Compiling the model...")
        logging.info("Compiling the model...")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy', metrics.SparseF1Score(num_classes=num_classes, average='weighted')])
        logging.info("Model compiled successfully.")

        # Train the model
        logging.info("Training the model...")
        history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
        logging.info("Model training completed.")


        # Log metrics
        final_accuracy = history.history["accuracy"][-1]
        final_val_accuracy = history.history["val_accuracy"][-1]
        final_f1 = history.history["f1_score"][-1]
        final_val_f1 = history.history["val_f1_score"][-1]
        mlflow.log_metric("train_accuracy", final_accuracy)
        mlflow.log_metric("val_accuracy", final_val_accuracy)
        mlflow.log_metric("train_f1_score", final_f1)
        mlflow.log_metric("val_f1_score", final_val_f1)
        # Save the model architecture
        model_json = model.to_json()
        archi_path = os.path.join(tools.MODEL_DIR, "efficientNetB1_model.json")
        logging.info(f"Saving model architecture to {archi_path}")
        if os.path.exists(tools.MODEL_DIR) is False:
            os.makedirs(tools.MODEL_DIR)
        with open(archi_path, "w") as json_file:
            json_file.write(model_json)
        logging.info("Model architecture saved successfully.")  

        # Save the model weights

        weights_path = os.path.join(tools.MODEL_DIR, "efficientNetB1_model.weights.h5")
        logging.info(f"Saving model weights to {weights_path}")
        model.save_weights(weights_path)
        logging.info("Model weights saved successfully.")

        # Log the training history
        logging.info("Logging training history...")
        history_path = os.path.join(tools.MODEL_DIR, "training_history_image.json")
        with open(history_path, "w") as json_file:
            json.dump(history.history, json_file)
        logging.info("Training history logged successfully.")

        mlflow.log_artifact(archi_path)
        mlflow.log_artifact(weights_path)
        mlflow.log_artifact(history_path)
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("src/models/models.py")
        logging.info("Artifacts logged to MLflow.")
        logging.info("MLflow run completed successfully.")

if __name__ == "__main__":
    main()
else:
    import inspect

    # If this script is imported, run the main function only if it is called from train_model.py
    stack = inspect.stack()
    for frame in stack:
        if "train_model_mlflow.py" in frame.filename:
            main()
            break    