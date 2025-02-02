import os
import json

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
#tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("state_size", 10, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_integer("max_question_length", 20, "Max length of the questions")
tf.app.flags.DEFINE_integer("max_context_length", 200, "Max length of the contexts")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def load_dataset(data_dir):
    train_ids_question_data_path = pjoin(FLAGS.data_dir, 'train.ids.question')
    train_ids_context_data_path = pjoin(FLAGS.data_dir, 'train.ids.context')    

    dataset = {}
    if tf.gfile.Exists(train_ids_question_data_path):
        train_question_ids = []
        train_question_lengths = []
        
        with tf.gfile.GFile(train_ids_question_data_path, mode="rb") as f:
            for line in f:
                line_ids = map(int, line.strip('\n').split(' '))
                line_length = len(line_ids)
                padding_length = FLAGS.max_question_length - line_length
                train_question_ids.append((line_ids + [0] * padding_length)[0:FLAGS.max_question_length])
                question_seq_length = min(line_length, FLAGS.max_question_length)
                train_question_lengths.append(question_seq_length)

            dataset['train_question_ids'] = train_question_ids
            dataset['train_question_lengths'] = train_question_lengths

    if tf.gfile.Exists(train_ids_context_data_path):
        train_context_ids = []
        train_context_lengths = []
        
        with tf.gfile.GFile(train_ids_context_data_path, mode="rb") as f:
            for line in f:
                c_line_ids = map(int, line.strip('\n').split(' '))
                c_line_length = len(c_line_ids)
                c_padding_length = FLAGS.max_context_length - line_length
                train_context_ids.append((c_line_ids + [0] * c_padding_length)[0:FLAGS.max_context_length])
                context_seq_length = min(c_line_length, FLAGS.max_context_length)
                train_context_lengths.append(context_seq_length)
            
            dataset['train_context_ids'] = train_context_ids
            dataset['train_context_lengths'] = train_context_lengths
            
    return dataset


def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    training_question_data_path = pjoin(FLAGS.data_dir, 'train.question')
    dataset = load_dataset(FLAGS.data_dir)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    pretrained_embeddings = np.load(embed_path)['glove']
    encoder = Encoder(size=FLAGS.state_size,
                      vocab_dim=FLAGS.embedding_size,
                      pretrained_embeddings = pretrained_embeddings,
                      max_question_length = FLAGS.max_question_length,
                      max_context_length = FLAGS.max_context_length)
    decoder = Decoder(output_size=FLAGS.output_size,
                      size = FLAGS.state_size,
                      max_context_length = FLAGS.max_context_length)

    qa = QASystem(encoder, decoder)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset, save_train_dir)

        qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
