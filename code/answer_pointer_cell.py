import tensorflow as tf
import numpy as np


class AnswerPointerCell(tf.contrib.rnn.RNNCell):
    """Answer Pointer Cell
    """
    def __init__(self, state_size, encodings, encodings_mask, max_num_context_tokens):
        self.num_units = state_size
        self.max_num_context_tokens = max_num_context_tokens
        self._state_size = tf.contrib.rnn.LSTMStateTuple(state_size, state_size)
        self._output_size = self.max_num_context_tokens

        self.encodings = encodings
        self.encodings_mask = encodings_mask
        
        self.initializer = tf.orthogonal_initializer()
            
        self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units = state_size,
                                                 initializer = self.initializer)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.

        The cell equations are

        G_t = tanh(W_q * H_q + (W_p * h_p_t + W_r * h_{t-1} + b_p))
        a_t = softmax(w_a * G_t + b_a)
        z_t = concat(h_p_t, H_q * a_t)
        h_t = LSTM.call(input = z_t, state = h_{t-1})

        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            xavier_initializer = tf.contrib.layers.xavier_initializer()
            zeros_initializer = tf.zeros_initializer()

            V = tf.get_variable(name = 'V', shape = [2 * self.num_units, self.num_units], dtype = tf.float64, initializer = xavier_initializer)
            W = tf.get_variable(name = 'W', shape = [self.num_units, self.num_units], dtype = tf.float64, initializer = xavier_initializer)
            b = tf.get_variable(name = 'b', shape = [1, self.num_units], dtype = tf.float64, initializer = zeros_initializer)

            H_ = tf.reshape(self.encodings, [-1, 2 * self.num_units])                                                                    # Dimensions = [Batch Size * (P + 1) x (2 * L)]

            F_k = tf.tanh(tf.matmul(H_, V) + tf.reshape(tf.tile(tf.matmul(state.h, W) + b, [1, self.max_num_context_tokens]), [-1, self.num_units]))  # Dimensions = [Batch Size * (P + 1) x L]

            v = tf.get_variable(name = 'v', shape = [self.num_units, 1], dtype = tf.float64, initializer = xavier_initializer)
            c = tf.get_variable(name = 'c', shape = [1,], dtype = tf.float64, initializer = zeros_initializer)
            beta_k_ = tf.matmul(F_k, v) + c                                                                                                   # Dimensions = [Batch Size * (P + 1) x 1]
            beta_k_ = tf.reshape(beta_k_, [-1, self.max_num_context_tokens])                                                                  # Dimensions = [Batch Size x (P + 1)]
            beta_k_ = tf.add(beta_k_, self.encodings_mask)
            beta_k = tf.nn.softmax(beta_k_)                                                                                                   # Dimensions = [Batch Size x (P + 1)]

            weighted_encodings = tf.reduce_sum(tf.reshape(tf.multiply(H_, tf.reshape(beta_k, [-1, 1])), [-1, self.max_num_context_tokens, 2 * self.num_units]), 1)   # Dimensions = [Batch Size X (2 * L)]
            output, new_state = self.lstm_cell(weighted_encodings, state, scope = scope)

        return beta_k, new_state
    


def do_answer_pointer_cell_test():
    with tf.Graph().as_default():
        with tf.variable_scope("test_answer_pointer_cell"):
            state_size = 3
            max_answer_length = 1
            max_num_context_tokens = 2
            fake_inputs = tf.fill(dims = (2, max_answer_length, 1), value = 0)
            encodings_placeholder = tf.placeholder(tf.float64, shape=(None, 2, 2 * state_size))
            encodings_mask_placeholder = tf.placeholder(tf.float64, shape=(None, 2))

            cell = AnswerPointerCell(state_size, encodings_placeholder, encodings_mask_placeholder, max_num_context_tokens)

            probabilities, final_state = tf.nn.dynamic_rnn(cell = cell,
                                                           inputs = fake_inputs,
                                                           dtype = tf.float64,
                                                           scope = 'answer_pointer')
            
            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                encodings = np.array([
                    [[0.4, 0.5, 0.6, 0.2, 0.5, 0.1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    [[0.3, -0.2, -0.1, 0.7, -0.3, -0.7], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float64)
                encodings_mask = np.array([[0.0, -np.inf],
                                           [0.0, 0.0]],
                                          dtype = np.float64)
                probabilities, final_state = session.run([probabilities, final_state], feed_dict={encodings_placeholder: encodings,
                                                                                                  encodings_mask_placeholder: encodings_mask})
                print("probabilities = " + str(probabilities))
                print("final_state = " + str(final_state))


if __name__ == "__main__":
    do_answer_pointer_cell_test()
