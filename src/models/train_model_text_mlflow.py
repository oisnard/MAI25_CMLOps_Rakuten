import tensorflow as tf
import pandas as pd
import src.tools.tools as tools 
import src.models.models as models
from src.models.metrics import SparseF1Score
from src.tools.datastream_mngt import get_current_type_datastream
import os 
import logging 
import mlflow
#import mlflow.tensorflow # --> A retirer, génère des incompatibilités entre keras 3.x et transformers



def main(pipeline_mode: str ='full'):
    logging.basicConfig(level=logging.INFO)

    logging.info("Available GPUs: %s", tf.config.list_physical_devices('GPU'))
    logging.info(f"Pipeline mode selected: {pipeline_mode}")
    params = tools.load_dataset_params_from_yaml()
    DATA_RATIO = params['data_ratio']
    MAX_LEN = params['models_parameters']['Camembert']['max_length']
    MODEL_NAME = params['models_parameters']['Camembert']['model_name']
    BATCH_SIZE = params['training_parameters']['batch_size']
    EPOCHS = params['training_parameters']['epochs']

    logging.info("Loading datasets...")
    X_train, X_val, y_train, y_val = tools.load_datasets()    
    logging.info(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")

    logging.info("Setting experiment...")
    mlflow.set_experiment("Rakuten_Text_Model")
    
    # Start MLflow run
    logging.info("Starting MLflow run...")
#    mlflow.tensorflow.autolog()  # A éviter - génère des incompatibilités entre keras 3.x et transformers
    with mlflow.start_run() as run:
        logging.info(f"MLflow run started. Run ID: {run.info.run_id}")
        mlflow.log_param("data_ratio", DATA_RATIO)
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_length", MAX_LEN)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)

        encode = models.load_encode_text_function(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train.feature.tolist(), y_train.prdtypecode.tolist()))
        train_dataset = train_dataset.map(encode).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val.feature.tolist(), y_val.prdtypecode.tolist()))
        val_dataset = val_dataset.map(encode).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        model = models.build_cam_model(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN, NUM_CLASSES=tools.NUM_CLASSES)
        weights_file = os.path.join(tools.MODEL_DIR, "camembert_model.weights.h5")
        if  'full' not in pipeline_mode and os.path.exists(weights_file):
            current_type_stream = get_current_type_datastream()
            if current_type_stream == "stream":
                try:
                    logging.info("Loading model weights from file for datastream mode...")
                    model.load_weights(os.path.join(tools.MODEL_DIR, "camembert_model.weights.h5"))
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
        mlflow.log_artifact("src/models/models.py")
        logging.info("Artifacts logged to MLflow.")



if __name__ == "__main__":
    main()
