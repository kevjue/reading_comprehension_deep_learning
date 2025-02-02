import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
import utils
import match_lstm_cell
import answer_pointer_cell

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, pretrained_embeddings, max_question_length, max_context_length, initialize_with_one = False):
        self.size = size
        self.pretrained_embeddings = pretrained_embeddings
        self.question_max_length = max_question_length
        self.context_max_length = max_context_length

        # This flag is used mostly for testing
        self.initialize_with_one = initialize_with_one

        self.encodings = None
        self.context_lengths_placeholder = None
        self.encoder_graph = self._build_encoder_graph()


    def _build_encoder_graph(self):
        with tf.Graph().as_default() as encoder_graph:
            self.question_ids_placeholder = tf.placeholder(tf.int32, shape = (None, self.question_max_length), name = 'question_ids_placeholder')
            self.question_lengths_placeholder = tf.placeholder(tf.int32, shape = (None,), name = 'question_lengths_placeholder')
            question_embeddings = tf.nn.embedding_lookup(self.pretrained_embeddings, self.question_ids_placeholder)

            if self.initialize_with_one:
                initializer = tf.ones_initializer()
            else:
                initializer = tf.orthogonal_initializer()

            # Create LSTM sequence for the question
            question_lstm_cell = tf.contrib.rnn.LSTMCell(num_units = self.size,
                                                         initializer = initializer)

            question_word_encodings, _ = tf.nn.dynamic_rnn(cell = question_lstm_cell,
                                                           dtype = tf.float64,
                                                           sequence_length = self.question_lengths_placeholder,
                                                           inputs = question_embeddings,
                                                           scope = 'question_rnn')

            # Create LSTM sequence for the context paragraph
            self.context_ids_placeholder = tf.placeholder(tf.int32, shape = (None, self.context_max_length), name = 'context_ids_placeholder')
            self.context_lengths_placeholder = tf.placeholder(tf.int32, shape = (None,), name = 'context_lengths_placeholder')
            context_embeddings =  tf.nn.embedding_lookup(self.pretrained_embeddings, self.context_ids_placeholder)

            # Create LSTM sequence for the question
            context_lstm_cell = tf.contrib.rnn.LSTMCell(num_units = self.size,
                                                        initializer = initializer)

            context_word_encodings, _ = tf.nn.dynamic_rnn(cell = context_lstm_cell,
                                                          dtype = tf.float64,
                                                          sequence_length = self.context_lengths_placeholder,
                                                          inputs = context_embeddings,
                                                          scope = 'context_rnn')

            # Create Match LSTM sequence for the context (combination of the context token and attention weighted question for that token)
            mlstm_cell_fw = match_lstm_cell.MatchLSTMCell(state_size = self.size,
                                                          question_vector = question_word_encodings,
                                                          question_mask = utils.create_softmax_mask(self.question_lengths_placeholder, self.question_max_length),
                                                          max_question_length = self.question_max_length,
                                                          initializer = initializer)

            mlstm_cell_bw = match_lstm_cell.MatchLSTMCell(state_size = self.size,
                                                          question_vector = question_word_encodings,
                                                          question_mask = utils.create_softmax_mask(self.question_lengths_placeholder, self.question_max_length),
                                                          max_question_length = self.question_max_length,
                                                          initializer = initializer)

            match_lstm_encodings, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = mlstm_cell_fw,
                                                                      cell_bw = mlstm_cell_bw,
                                                                      dtype = tf.float64, 
                                                                      sequence_length = self.context_lengths_placeholder,
                                                                      inputs = context_word_encodings,
                                                                      scope = 'match_lstm_birnn')

            self.encodings = tf.concat(values = [match_lstm_encodings[0], match_lstm_encodings[1]], axis = 2, name = 'encodings')
        return encoder_graph

    
    def encode(self, dataset, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        
        feed_dict = {self.question_ids_placeholder: dataset['train_question_ids'],
                     self.question_lengths_placeholder: dataset['train_question_lengths'],
                     self.context_ids_placeholder: dataset['train_context_ids'],
                     self.context_lengths_placeholder: dataset['train_context_lengths']}

        with tf.Session(graph = self.encoder_graph) as sess:
            sess.run(tf.global_variables_initializer())
            outputs = sess.run(self.encodings, feed_dict=feed_dict)
        
        return outputs

    

class Decoder(object):
    def __init__(self, output_size, size, max_context_length, max_answer_length):
        self.size = size
        self.max_num_context_tokens = max_context_length + 1
        self.max_context_length = max_context_length
        self.max_answer_length = max_answer_length

        self.decoder_graph = self._build_decoder_graph()


    def _build_decoder_graph(self):
        with tf.Graph().as_default() as decoder_graph:
            self.encodings_placeholder = tf.placeholder(tf.float64, shape = (None, self.max_context_length, 2 * self.size), name = 'encodings_placeholder')
            self.encodings_lengths_placeholder = tf.placeholder(tf.int32, shape = (None,), name = 'encodings_length_placeholder')

            # Add the zero vector to the encodings (for the end of answer token)
            batch_size = tf.shape(self.encodings_placeholder)[0]
            zero_vector = tf.fill(dims = (batch_size, 1, 2 * self.size), value = np.float64(0.0))
            encodings = tf.concat([self.encodings_placeholder, zero_vector], 1)
            encodings_length = self.encodings_lengths_placeholder + 1

            ap_cell = answer_pointer_cell.AnswerPointerCell(state_size = self.size,
                                                            encodings = encodings,
                                                            encodings_mask = utils.create_softmax_mask(encodings_length, self.max_num_context_tokens),
                                                            max_num_context_tokens = self.max_num_context_tokens)

            # dynamic_rnn function requires an input tensor.  The anwer pointer layer doesn't require any inputs (other than the encoded
            # context and question),  so we need to generate a fake input tensor.
            fake_inputs = tf.fill(dims = (batch_size, self.max_answer_length, 1), value = 0)
            answer_softmaxes, _ = tf.nn.dynamic_rnn(cell = ap_cell,
                                                    dtype = tf.float64,
                                                    inputs = fake_inputs,
                                                    scope = 'ap_rnn')

            # Need to create a graph label for the answer_softmax computation node.  The tf.nn.dynamic_rnn function doesn't allow
            # for setting that label, so I'm using the tf.identify function
            self.answer_softmaxes = tf.identity(answer_softmaxes, 'answer_softmaxes')

        return decoder_graph



    def decode(self, knowledge_rep, knowledge_rep_lengths):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        feed_dict = {self.encodings_placeholder: knowledge_rep,
                     self.encodings_lengths_placeholder: knowledge_rep_lengths}

        with tf.Session(graph = self.decoder_graph) as sess:
            sess.run(tf.global_variables_initializer())
            outputs = sess.run(self.answer_softmaxes, feed_dict=feed_dict)
        
        return outputs

    
    
class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder
        self.graph = tf.Graph()

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            #self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        with self.graph.as_default():
            self.question_ids_placeholder = tf.placeholder(tf.int32, shape = (None, self.question_max_length), name = 'question_ids_placeholder')
            self.question_lengths_placeholder = tf.placeholder(tf.int32, shape = (None,), name = 'question_lengths_placeholder')
            self.context_ids_placeholder = tf.placeholder(tf.int32, shape = (None, self.context_max_length), name = 'context_ids_placeholder')
            self.context_lengths_placeholder = tf.placeholder(tf.int32, shape = (None,), name = 'context_lengths_placeholder')
            
            encodings, = tf.import_graph_def(self.encoder.encoder_graph.as_graph_def(),
                                             input_map = {'question_ids_placeholder:0': self.question_ids_placeholder,
                                                          'question_lengths_placeholder:0': self.question_lengths_placeholder,
                                                          'context_ids_placeholder:0': self.context_ids_placeholder,
                                                          'context_lengths_placeholder:0': self.context_lengths_placeholder},
                                             return_elements = ['encodings:0'])

            self.answer_softmaxes, = tf.import_graph_def(self.decoder.decoder_graph.as_graph_def(),
                                                         input_map = {'encodings_placeholder:0', encodings,
                                                                      'encodings_length_placeholder:0', self.context_lenghts_placeholder},
                                                         return_elements = ['answer_softmaxes'])
        
        
    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with self.graph.as_default():
            

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """

        with vs.variable_scope("embeddings"):
            pass
            
        

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em



    def run_epoch(self, session, training_data):
        encoded_input = self.encoder.encode(session,
                                            training_data,
                                            None)

        import numpy
        numpy.set_printoptions(threshold=numpy.nan)
        
        print encoded_input
        sys.exit(0)
        
    
    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        for epoch in range(100):
            print "running epoch #%d" % epoch
            self.run_epoch(session, dataset)

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))



def run_encoder_tests(max_context_length, size):
    test_pretrained_embeddings = np.array([[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.044457, -0.49688, -0.17862, -0.00066023, -0.6566],
                                           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709, -0.42852, -0.55641, -0.364, -0.23938],
                                           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973, -0.43478, -0.31086, -0.44999, -0.29486],
                                           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603, 0.18157, -0.52393, 0.10381, -0.17566],
                                           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246, -0.41376, 0.13228, -0.29847, -0.085253]],
                                          dtype = np.float64)
        
    test_encoder = Encoder(size = size,
                           pretrained_embeddings = test_pretrained_embeddings,
                           max_context_length = max_context_length,
                           max_question_length = 5)

    question1_ids = [3, 2, 1, 1, 3]
    question2_ids = [3, 1, 3, 0, 0]
    question_lengths = [5, 3]
    question_masks = [[1, 1, 1, 1, 1],
                      [1, 1, 1, 0, 0]]

    context1_ids = [4, 4, 1, 2, 2, 4, 1, 3, 0, 0]
    context2_ids = [1, 4, 2, 3, 3, 1, 4, 1, 3, 1]
    context_lengths = [8, 10]
    context_masks = [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    training_data = {'train_question_ids': [question1_ids, question2_ids],
                     'train_question_lengths': question_lengths,
                     'train_question_masks': question_masks,
                     'train_context_ids': [context1_ids, context2_ids],
                     'train_context_lengths': context_lengths}

    encodings = test_encoder.encode(training_data,
                                    None)
    return encodings, context_lengths


def run_decoder_tests(encodings, encodings_lengths, max_context_length, size):
    test_decoder = Decoder(output_size = None,
                           size = size,
                           max_context_length = max_context_length,
                           max_answer_length = 5)

    answer_softmaxes = test_decoder.decode(encodings, encodings_lengths)
    return answer_softmaxes
    
    

if __name__ == '__main__':
    max_context_length = 10
    size = 10
    encodings, encodings_lengths = run_encoder_tests(max_context_length, size)
    print(encodings)
    answer_softmaxes = run_decoder_tests(encodings, encodings_lengths, max_context_length, size)
    print(answer_softmaxes)
