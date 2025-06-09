from transformers import TFCamembertModel, CamembertTokenizer
import tensorflow as tf

IMG_SIZE = 500


def load_encode_text_function(MODEL_NAME="almanach/camembert-base", MAX_LEN=32) -> callable:
    """
    Load the Camembert tokenizer and return a function.
    
    Args:
        MODEL_NAME (str): The name of the pre-trained Camembert model.
        MAX_LEN (int): The maximum length for padding/truncation.
   
    Returns:
        CamembertTokenizer: The loaded tokenizer.
    """
    tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
    def encode(text, label):
        def _tokenize(t, l):
            enc = tokenizer.encode_plus(
                t.numpy().decode("utf-8"),
                padding='max_length',
                truncation=True,
                max_length=MAX_LEN,
                return_tensors='np'
            )
            return enc["input_ids"][0], enc["attention_mask"][0], l
        input_ids, attention_mask, label = tf.py_function(
            func=_tokenize,
            inp=[text, label],
            Tout=(tf.int32, tf.int32, tf.int32)
        )
        input_ids.set_shape([MAX_LEN])
        attention_mask.set_shape([MAX_LEN])
        label.set_shape([])
        return {"input_ids": input_ids, "attention_mask": attention_mask}, label
    return encode

# Function to build the Camembert model for multi-class classification
def build_cam_model(MODEL_NAME="almanach/camembert-base", MAX_LEN=32, NUM_CLASSES=27) -> tf.keras.Model:
    """
    Build a Camembert model for multi-class classification.
    Args:
        MODEL_NAME (str): The name of the pre-trained Camembert model.
        MAX_LEN (int): The maximum length for padding/truncation.
        NUM_CLASSES (int): The number of classes for classification.
    Returns:
        tf.keras.Model: The constructed Camembert model.
    """
    input_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")

    camembert = TFCamembertModel.from_pretrained(
        MODEL_NAME,
        output_attentions=False,
        output_hidden_states=False
    )

    camembert.roberta.pooler.dense.trainable = False
    camembert.roberta.pooler.dense.kernel.assign(tf.zeros_like(camembert.roberta.pooler.dense.kernel))
    camembert.roberta.pooler.dense.bias.assign(tf.zeros_like(camembert.roberta.pooler.dense.bias))


    outputs = camembert(input_ids=input_ids, attention_mask=attention_mask)
    x = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
    x = tf.reshape(x, (-1, MAX_LEN * 768))  # aplatissement
    x = tf.keras.layers.Dense(768, activation="tanh")(x)
    logits = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)
    return model





# Define a function to load and preprocess an image based on its file path
def load_and_preprocess_image_from_path(filepath):
    """
    Load and preprocess an image from a file path.
    Args:
        filepath (tf.Tensor): A tensor containing the file path of the image.
    Returns:
        tf.Tensor: A preprocessed image tensor.
    """
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image

# Build an model based on EfficientNetB1
def build_model_image_efficientNetB1(num_classes: int, nb_trainable_layers: int) -> tf.keras.Model:
    """
    Build a model based on EfficientNetB1 for image classification.
    Args:
        num_classes (int): The number of classes for classification.
        nb_trainable_layers (int): The number of trainable layers in the base model.
    Returns:
        tf.keras.Model: The constructed EfficientNetB1 model.
    """
    # Define the input layer for image paths
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)
 
    # Use tf.map_fn to apply the image loading and preprocessing function to each path in the batch
    def load_and_preprocess_batch(paths):
        return tf.map_fn(
            load_and_preprocess_image_from_path,
            paths,
            fn_output_signature=tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
        )

    # Lambda avec forme de sortie spécifiée
    images = tf.keras.layers.Lambda(
        load_and_preprocess_batch,
        output_shape=(IMG_SIZE, IMG_SIZE, 3)
    )(inputs)


    base_model = tf.keras.applications.EfficientNetB1(weights='imagenet', 
                                                      include_top=False, 
                                                      input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    if nb_trainable_layers > 0:
        for layer in base_model.layers[-nb_trainable_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    x = base_model(images) #, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(units=1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model