import tensorflow as tf

class MLSTM_Cell(tf.contrib.rnn.RNNCell):
    def __init__(self, input_size, state_size, question_hidden_vectors):
        self.input_size = input_size
        self._state_size = state_size
        self.question_hidden_vectors = question_hidden_vectors


    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the mLSTM equations are:

        G_t = tanh(W_q question_hidden_vectors + (x_t W_p + h_{t-1} W_r + b_p) x Q)
        a_t = sotfmax(w G_t + b x Q)
        z_t = concat(x_t, question_hidden_vectors a)
        h_t = LSTM(z_t)

        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__
        
        return output, new_state

