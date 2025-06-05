import tensorflow as tf
import pandas as pd
import src.tools.tools as tools 
import src.models.models as models
from src.models.metrics import SparseF1Score
import os 
import logging 
import mlflow
import mlflow.tensorflow

def load_datasets() -> tuple:
    """
    Function to load the datasets for training and validation.
    Returns:
        X_train, y_train, X_val, y_val : DataFrames containing the training and validation data.
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

    if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
        logging.error("One or more datasets are empty.")
        raise ValueError("One or more datasets are empty.")

    if not X_train.index.equals(y_train.index) or not X_val.index.equals(y_val.index):
        logging.error("Indices of features and labels do not match.")
        raise ValueError("Indices of features and labels do not match.")

    if 'feature' not in X_train.columns or 'feature' not in X_val.columns:
        logging.error("The 'feature' column is missing in one of the datasets.")
        raise ValueError("The 'feature' column is missing in one of the datasets.")

    if 'prdtypecode' not in y_train.columns or 'prdtypecode' not in y_val.columns:
        logging.error("The 'prdtypecode' column is missing in one of the label datasets.")
        raise ValueError("The 'prdtypecode' column is missing in one of the label datasets.")

    if not pd.api.types.is_integer_dtype(y_train['prdtypecode']) or not pd.api.types.is_integer_dtype(y_val['prdtypecode']):
        logging.error("The 'prdtypecode' column must be of integer type.")
        raise ValueError("The 'prdtypecode' column must be of integer type.")

    if not pd.api.types.is_string_dtype(X_train['feature']) or not pd.api.types.is_string_dtype(X_val['feature']):
        logging.error("The 'feature' column must be of string type.")
        raise ValueError("The 'feature' column must be of string type.")

    if y_train['prdtypecode'].isnull().any() or y_val['prdtypecode'].isnull().any():
        logging.error("The 'prdtypecode' column contains null values.")
        raise ValueError("The 'prdtypecode' column contains null values.")

    if X_train['feature'].isnull().any() or X_val['feature'].isnull().any():
        logging.error("The 'feature' column contains null values.")
        raise ValueError("The 'feature' column contains null values.")

    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    params = tools.load_dataset_params_from_yaml()
    MAX_LEN = params['models_parameters']['camembert']['max_length']
    MODEL_NAME = params['models_parameters']['camembert']['model_name']
    BATCH_SIZE = params['training_parameters']['batch_size']
    EPOCHS = params['training_parameters']['epochs']

    logging.info("Loading datasets...")
    X_train, X_val, y_train, y_val = load_datasets()    
    logging.info(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")

    mlflow.set_experiment("Rakuten_Text_Model")
    with mlflow.start_run():
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_length", MAX_LEN)
        mlflow.log_param("batch_size", BATCH_SIZE)

        encode = models.load_encode_text_function(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train.feature.tolist(), y_train.prdtypecode.tolist()))
        train_dataset = train_dataset.map(encode).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val.feature.tolist(), y_val.prdtypecode.tolist()))
        val_dataset = val_dataset.map(encode).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        model = models.build_cam_model(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN, NUM_CLASSES=tools.NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy", SparseF1Score(num_classes=tools.NUM_CLASSES, average='weighted')]
        )

        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=val_dataset
        )

        logging.info("Training completed.")

        final_accuracy = history.history["accuracy"][-1]
        final_val_accuracy = history.history["val_accuracy"][-1]
        final_f1 = history.history["f1_score"][-1]
        final_val_f1 = history.history["val_f1_score"][-1]

        mlflow.log_metric("train_accuracy", final_accuracy)
        mlflow.log_metric("val_accuracy", final_val_accuracy)
        mlflow.log_metric("train_f1_score", final_f1)
        mlflow.log_metric("val_f1_score", final_val_f1)

        weights_path = os.path.join(tools.MODEL_DIR, "camembert_model.weights.h5")
        archi_path = os.path.join(tools.MODEL_DIR, "camembert_model.json")

        model.save_weights(weights_path)
        model_json = model.to_json()
        with open(archi_path, "w") as json_file:
            json_file.write(model_json)

        logging.info("Model saved locally.")

        mlflow.log_artifact(weights_path)
        mlflow.log_artifact(archi_path)
        mlflow.log_artifact("params.yaml")
