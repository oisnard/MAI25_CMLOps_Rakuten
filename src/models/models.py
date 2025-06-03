from transformers import TFCamembertModel, CamembertTokenizer
import tensorflow as tf


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



