import tensorflow as tf
import numpy as np

def create_softmax_mask(batch_seq_lengths, max_seq_length):
    seq_length_transposed = tf.expand_dims(batch_seq_lengths, 1)
    range = tf.range(0, max_seq_length)
    range_row = tf.expand_dims(range, 0)
    
    softmax_mask_condition = tf.less(seq_length_transposed, range_row)
    softmax_mask = tf.where(softmax_mask_condition,
                            tf.fill(tf.shape(softmax_mask_condition), value=np.float64(-np.inf)),
                            tf.fill(tf.shape(softmax_mask_condition), value=np.float64(0)))
    return softmax_mask

