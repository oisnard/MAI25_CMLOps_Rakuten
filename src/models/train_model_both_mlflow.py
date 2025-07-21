import tensorflow as tf
import pandas as pd
import src.tools.tools as tools 
import src.models.models as models
from src.models.metrics import SparseF1Score
from src.tools.datastream_mngt import get_current_type_datastream
import os 
import logging 
import mlflow



def main(pipeline_mode: str ='full'):
    logging.basicConfig(level=logging.INFO)

    logging.info("Available GPUs: %s", tf.config.list_physical_devices('GPU'))

    params = tools.load_dataset_params_from_yaml()
    DATA_RATIO = params['data_ratio']
    if not isinstance(DATA_RATIO, float) or not (0 < DATA_RATIO <= 1):
        logging.error(f"Invalid DATA_RATIO value: {DATA_RATIO}. It should be a float between 0 and 1.")
        raise ValueError(f"Invalid DATA_RATIO value: {DATA_RATIO}. It should be a float between 0 and 1.")

    MAX_LEN = params['models_parameters']['Camembert']['max_length']
    if not isinstance(MAX_LEN, int) or MAX_LEN <= 0:
        logging.error(f"Invalid MAX_LEN value: {MAX_LEN}. It should be a positive integer.")
        raise ValueError(f"Invalid MAX_LEN value: {MAX_LEN}. It should be a positive integer.")

    MODEL_NAME = params['models_parameters']['Camembert']['model_name']
    if not isinstance(MODEL_NAME, str) or not MODEL_NAME:
        logging.error(f"Invalid MODEL_NAME value: {MODEL_NAME}. It should be a non-empty string.")
        raise ValueError(f"Invalid MODEL_NAME value: {MODEL_NAME}. It should be a non-empty string.")

    BATCH_SIZE = params['training_parameters']['batch_size']
    if not isinstance(BATCH_SIZE, int) or BATCH_SIZE <= 0:
        logging.error(f"Invalid BATCH_SIZE value: {BATCH_SIZE}. It should be a positive integer.")
        raise ValueError(f"Invalid BATCH_SIZE value: {BATCH_SIZE}. It should be a positive integer.")

    EPOCHS = params['training_parameters']['epochs']
    if not isinstance(EPOCHS, int) or EPOCHS <= 0:
        logging.error(f"Invalid EPOCHS value: {EPOCHS}. It should be a positive integer.")
        raise ValueError(f"Invalid EPOCHS value: {EPOCHS}. It should be a positive integer.")

    nb_trainable_layers = params['models_parameters']['EfficientNetB1']['nb_trainable_layers']
    if not isinstance(nb_trainable_layers, int) or nb_trainable_layers < 0:
        logging.error(f"Invalid nb_trainable_layers value: {nb_trainable_layers}. It should be a non-negative integer.")
        raise ValueError(f"Invalid nb_trainable_layers value: {nb_trainable_layers}. It should be a non-negative integer.")
    

    logging.info("Loading datasets...")
    X_train, X_val, y_train, y_val = tools.load_datasets()    
    logging.info(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")

    logging.info("Setting experiment...")
    mlflow.set_experiment("Rakuten_Text_and_Image_Model")
    
    # Start MLflow run
    logging.info("Starting MLflow run...")
#    mlflow.tensorflow.autolog()  # A éviter - génère des incompatibilités entre keras 3.x et transformers
    with mlflow.start_run() as run:
        logging.info(f"MLflow run started. Run ID: {run.info.run_id}")
        mlflow.log_param("data_ratio", DATA_RATIO)
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_type", "EfficientNetB1")
        mlflow.log_param("nb_trainable_layers", nb_trainable_layers)
        mlflow.log_param("max_length", MAX_LEN)
        mlflow.log_param("batch_size", BATCH_SIZE)

        mlflow.log_param("epochs", EPOCHS)

        encode = models.load_encode_data(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train.feature.tolist(), 
                                                            X_train.image_path.tolist(),
                                                            y_train.prdtypecode.tolist()))
        train_dataset = train_dataset.map(encode).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val.feature.tolist(), 
                                                           X_val.image_path.tolist(),
                                                           y_val.prdtypecode.tolist()))
        val_dataset = val_dataset.map(encode).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        model = models.build_merged_model(MODEL_NAME=MODEL_NAME, 
                                          MAX_LEN=MAX_LEN, 
                                          NUM_CLASSES=tools.NUM_CLASSES,
                                          nb_trainable_layers=nb_trainable_layers)
        weights_file = os.path.join(tools.MODEL_DIR, "merged_model.weights.h5")
        if  'full' not in pipeline_mode and os.path.exists(weights_file):
            if "stream" in get_current_type_datastream():
                try:
                    logging.info("Loading model weights from file for datastream mode...")
                    model.load_weights(os.path.join(tools.MODEL_DIR, "merged_model.weights.h5"))
                    logging.info("Model weights loaded successfully.")
                except Exception as e:
                    logging.error(f"Error loading model weights: {e}")
                    raise e
        # Compile the model
        logging.info("Compiling the model...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy", SparseF1Score(num_classes=tools.NUM_CLASSES, average='weighted')]
        )

        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=val_dataset,
            verbose=1
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

        if os.path.exists(tools.MODEL_DIR) is False:
            os.makedirs(tools.MODEL_DIR)
        weights_path = os.path.join(tools.MODEL_DIR, "merged_model.weights.h5")
        archi_path = os.path.join(tools.MODEL_DIR, "merged_model.json")

        model.save_weights(weights_path)
        model_json = model.to_json()
        with open(archi_path, "w") as json_file:
            json_file.write(model_json)

        logging.info("Model saved locally.")

        mlflow.log_artifact(weights_path)
        mlflow.log_artifact(archi_path)
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("src/models/models.py")
        logging.info("Artifacts logged to MLflow.")



if __name__ == "__main__":
    main()
