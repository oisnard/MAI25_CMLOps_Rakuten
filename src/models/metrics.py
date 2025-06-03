import tensorflow as tf

# Define a custom metric for sparse F1 score 
# This metric is designed for multi-class classification problems
# where the labels are sparse (i.e., each sample belongs to one class).
# tf.keras.metrics.F1Score needs 

class SparseF1Score(tf.keras.metrics.Metric):
    """ 
    A specific Class F1Score is defined (not available in tensorflow 2.10)
    """
    def __init__(self, num_classes=27, average='macro', name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.true_positives = self.add_weight(name='tp', initializer='zeros', shape=(num_classes,))
        self.false_positives = self.add_weight(name='fp', initializer='zeros', shape=(num_classes,))
        self.false_negatives = self.add_weight(name='fn', initializer='zeros', shape=(num_classes,))
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)
        y_pred_one_hot = tf.one_hot(y_pred, depth=self.num_classes)
        true_positives = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        false_positives = tf.reduce_sum((1 - y_true_one_hot) * y_pred_one_hot, axis=0)
        false_negatives = tf.reduce_sum(y_true_one_hot * (1 - y_pred_one_hot), axis=0)
        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        if self.average == 'macro':
            return tf.reduce_mean(f1)
        elif self.average == 'weighted':
            weights = self.true_positives + self.false_negatives
            weights = weights / tf.reduce_sum(weights)
            return tf.reduce_sum(f1 * weights)
        else:  # 'micro'
            precision = tf.reduce_sum(self.true_positives) / (tf.reduce_sum(self.true_positives) + tf.reduce_sum(self.false_positives) + tf.keras.backend.epsilon())
            recall = tf.reduce_sum(self.true_positives) / (tf.reduce_sum(self.true_positives) + tf.reduce_sum(self.false_negatives) + tf.keras.backend.epsilon())
            return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))


