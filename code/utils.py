def create_softmax_mask(batch_seq_lengths, max_seq_length):
    seq_length_transposed = tf.expand_dims(batch_seq_lengths, 1)
    range = tf.range(0, max_seq_length)
    range_row = tf.expand_dims(range, 0)
    
    softmax_mask = tf.less(seq_length_transposed, range_row)
    tf.assign(softmax_mask, tf.where(softmax_mask, tf.constant(-np.inf, shape = softmax_mask.shape()), tf.zeros_like(softmax_mask)))
    return softmax_mask

