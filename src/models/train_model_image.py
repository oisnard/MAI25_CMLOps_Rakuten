import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1 
import os 
import logging 
import src.tools.tools as tools
import src.models.models as models
import src.models.metrics as metrics


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Training model based on images...")

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
    X_train, X_val, y_train, y_val = tools.load_datasets()    
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


    # Build the model
    logging.info("Building the EfficientNetB1 model...")
    model = models.build_model_image_efficientNetB1(num_classes=num_classes, nb_trainable_layers=nb_trainable_layers)

    logging.info("Model built successfully.")

    # Compile the model
    logging.info("Compiling the model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy', metrics.SparseF1Score(num_classes=num_classes, average='weighted')])
    logging.info("Model compiled successfully.")
    # Train the model
    logging.info("Training the model...")
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
    logging.info("Model training completed.")
    # Save the model
    if os.path.exists(tools.MODEL_DIR) is False:
        os.makedirs(tools.MODEL_DIR)
    model.save_weights(os.path.join(tools.MODEL_DIR, "efficientNetB1_model.weights.h5"))
    logging.info("Model weightssaved successfully.")
    # Save the model architecture
    model_json = model.to_json()
    with open(os.path.join(tools.MODEL_DIR, "efficientNetB1_model.json"), "w") as json_file:
        json_file.write(model_json)
    logging.info("Model architecture saved successfully.")
    # Log the training history
    for key, values in history.history.items():
        logging.info(f"Training history - {key}: {values}")
    # Log the final training and validation accuracy and F1 score
    final_accuracy = history.history["accuracy"][-1]
    final_val_accuracy = history.history["val_accuracy"][-1]
    final_f1 = history.history["f1_score"][-1]
    final_val_f1 = history.history["val_f1_score"][-1]
    logging.info(f"Final training accuracy: {final_accuracy}")
    logging.info(f"Final validation accuracy: {final_val_accuracy}")
    logging.info(f"Final training F1 score: {final_f1}")
    logging.info(f"Final validation F1 score: {final_val_f1}")

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