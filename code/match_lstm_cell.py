import tensorflow as tf
import numpy as np


class MatchLSTMCell(tf.contrib.rnn.RNNCell):
    """Match LSTM Cell
    """
    def __init__(self, state_size, question_vector, question_mask, max_question_length, initializer = None):
        self.num_units = state_size
        self._state_size = tf.contrib.rnn.LSTMStateTuple(state_size, state_size)
        self._output_size = state_size

        self.question_vector = question_vector
        self.question_mask = question_mask
        self.max_question_length = max_question_length

        if initializer:
            self.initializer = initializer
        else:
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

            W_q = tf.get_variable(name = 'W_q', shape = [self.num_units, self.num_units], dtype = tf.float64, initializer = xavier_initializer)
            W_p = tf.get_variable(name = 'W_p', shape = [self.num_units, self.num_units], dtype = tf.float64, initializer = xavier_initializer)
            W_r = tf.get_variable(name = 'W_r', shape = [self.num_units, self.num_units], dtype = tf.float64, initializer = xavier_initializer)
            b_p = tf.get_variable(name = 'b_p', shape = [1, self.num_units], dtype = tf.float64, initializer = zeros_initializer)

            Q_ = tf.reshape(self.question_vector, [-1, self.num_units])                                                                        # Dimensions = [Batch Size * Q x L]

            G_t = tf.tanh(tf.matmul(Q_, W_q) + tf.reshape(tf.tile(tf.matmul(inputs, W_p) + tf.matmul(state.h, W_r) + b_p, [1, self.max_question_length]), [-1, self.num_units]))   # Dimensions = [Batch Size * Q x L]

            w_a = tf.get_variable(name = 'w_a', shape = [self.num_units, 1], dtype = tf.float64, initializer = xavier_initializer)
            b_a = tf.get_variable(name = 'b_a', shape = [1,], dtype = tf.float64, initializer = zeros_initializer)
            a_t_ = tf.matmul(G_t, w_a) + b_a                                                                                                   # Dimensions = [Batch Size * Q x 1]
            a_t_ = tf.reshape(a_t_, [-1, self.max_question_length])                                                                            # Dimensions = [Batch Size x Q]
            a_t_ = tf.add(a_t_, self.question_mask)
            a_t = tf.nn.softmax(a_t_)                                                                                                          # Dimensions = [Batch Size x Q]

            weighted_questions = tf.reduce_sum(tf.reshape(tf.multiply(Q_, tf.reshape(a_t, [-1, 1])), [-1, self.max_question_length, self.num_units]), 1)   # Dimensions = [Batch Size X L]
            z_t = tf.concat([inputs, weighted_questions], 1)
            
            output, new_state = self.lstm_cell(z_t, state, scope = scope)

        return output, new_state
    


def do_match_lstm_cell_test():
    with tf.Graph().as_default():
        with tf.variable_scope("test_match_lstm_cell"):
            state_size = 3
            max_seq_length = 1
            input_placeholder = tf.placeholder(tf.float64, shape=(None, max_seq_length, state_size))
            h_placeholder = tf.placeholder(tf.float64, shape=(None, state_size))
            H_q_placeholder = tf.placeholder(tf.float64, shape=(None, 2, state_size))
            seq_length_placeholder = tf.placeholder(tf.int32, shape=(None,))

            #with tf.variable_scope("match_lstm"):
            #    tf.get_variable("W_q", initializer=np.array(np.eye(state_size, state_size), dtype=np.float64))
            #    tf.get_variable("W_p", initializer=np.array(np.eye(state_size, state_size), dtype=np.float64))
            #    tf.get_variable("W_r", initializer=np.array(np.eye(state_size, state_size), dtype=np.float64))
            #    tf.get_variable("b_p", initializer=np.array(np.ones([1, state_size]), dtype=np.float64))
            #    tf.get_variable("w_a", initializer=np.array(np.ones([state_size, 1]), dtype=np.float64))
            #    tf.get_variable("b_a", initializer=np.array(np.ones(1), dtype=np.float64))

            #tf.get_variable_scope().reuse_variables()
            cell = MatchLSTMCell(state_size, state_size, H_q_placeholder, state_size, 2)

            outputs, final_state = tf.nn.dynamic_rnn(cell = cell,
                                                     sequence_length = seq_length_placeholder,
                                                     inputs = input_placeholder,
                                                     dtype = tf.float64,
                                                     scope = 'match_lstm')
            
            #y_var, ht_var = cell(x_placeholder, h_placeholder, scope="match_lstm")

            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                lengths = np.array([1, 1], dtype=np.int32)
                x = np.array([
                    [[0.4, 0.5, 0.6]],
                    [[0.3, -0.2, -0.1]]], dtype=np.float64)
                h = np.array([
                    [0.2, 0.5, 0.7],
                    [-0.3, -0.3, -0.3]], dtype=np.float64)
                y = np.array([[0.4, 0.5, 0.6, 0.650, 0.199, 0.003],
                              [0.3, -0.2, -0.1, 0.7, 0.182, 0.135]],
                             dtype = np.float64)
                H_q = np.array([[[0.6, 0.3, -0.3], [0.7, 0.1, 0.3]],
                                [[0.7, -0.4, 0.2], [0.7, 0.5, 0.1]]])
                ht = y
                
                outputs, final_state = session.run([outputs, final_state], feed_dict={seq_length_placeholder: lengths,
                                                                                      input_placeholder: x,
                                                                                      H_q_placeholder: H_q})
                print("outputs = " + str(outputs))
                print("final_state = " + str(final_state))

                #assert np.allclose(y_, ht_), "output and state should be equal."
                #assert np.allclose(ht, ht_, atol=1e-2), "new state vector does not seem to be correct."


if __name__ == "__main__":
    do_match_lstm_cell_test()
    
                                
