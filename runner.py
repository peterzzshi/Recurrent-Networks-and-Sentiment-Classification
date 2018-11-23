import numpy as np
import tensorflow as tf
import re
from random import randint
import datetime
import os
from pathlib import Path
import pickle as pk
import glob

import argparse


BATCH_SIZE = 64
MAX_WORDS_IN_REVIEW = 200  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

SAVE_FREQ = 100
iterations = 100000

checkpoints_dir = "./checkpoints"


def load_data(path='./data/train'):
    """
    Load raw reviews from text files, and apply preprocessing
    Append positive reviews first, and negative reviews second
    RETURN: List of strings where each element is a preprocessed review.
    """
    print("Loading IMDB Data...")
    data = []

    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir, path + '/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir, path + '/neg/*')))
    print("Parsing %s files" % len(file_list))
    for i, f in enumerate(file_list):
        with open(f, "r") as openf:
            s = openf.read()
            data.append(preprocess(s))  # NOTE: Preprocessing code called here on all reviews
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    If loaded for the first time, serialize the final dict for quicker loading.
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """

    emmbed_file = Path("./embeddings.pkl")
    if emmbed_file.is_file():
        # embeddings already serialized, just load them
        print("Local Embeddings pickle found, loading...")
        with open("./embeddings.pkl", 'rb') as f:
            return pk.load(f)
    else:
        # create the embeddings
        print("Building embeddings dictionary...")
        data = open("glove.6B.50d.txt", 'r', encoding="utf-8")
        embeddings = [[0] * EMBEDDING_SIZE]
        word_index_dict = {'UNK': 0}  # first row is for unknown words
        index = 1
        for line in data:
            splitLine = line.split()
            word = tf.compat.as_str(splitLine[0])
            embedding = [float(val) for val in splitLine[1:]]
            embeddings.append(embedding)
            word_index_dict[word] = index
            index += 1
        data.close()

        # pickle them
        with open('./embeddings.pkl', 'wb') as f:
            print("Creating local embeddings pickle for faster loading...")
            # Pickle the 'data' dictionary using the highest protocol available.
            pk.dump((embeddings, word_index_dict), f, pk.HIGHEST_PROTOCOL)

    return embeddings, word_index_dict


def embedd_data(training_data_text, e_arr, e_dict):
    """
    Take the list of strings created by load_data() and apply an
    embeddings lookup using the created embeddings array and dictionary
    RETURN: 3-D Numpy mat where axis 0 = reviews
    axis 1 = words in review
    axis 2 = emedding vec for word

    Note that the array then has the shape: NUM_SAMPLES x MAX_WORDS_IN_REVIEW x EMBEDDING_SIZE
    Zero pad embedding if sentence is shorter than MAX_WORDS_IN_REVIEW
    ensure
    """
    num_samples = len(training_data_text)
    embedded = np.zeros([num_samples, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE])
    for i in range(num_samples):
        review_mat = np.zeros([MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE])
        # Iterate to either the end of the sentence of the max num of words, whichever is less
        for w in range(min(len(training_data_text[i]), MAX_WORDS_IN_REVIEW)):
            # assign embedding of that word or to the UNK token if that word isn't in the dict
            review_mat[w] = e_arr[e_dict.get(training_data_text[i][w], 0)]
        embedded[i] = review_mat
    return embedded




def preprocess(review):
    """
    Apply preprocessing to a single review. 
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    processed_review = [r for r in re.sub('<.*?>|[?!-;,.()0-9]', '', review.lower()).split() if r not in stop_words]
    return processed_review




def define_graph():


    lstm_size = 100
    n_classes = 2
    # num_layers = 3

    input_data = tf.placeholder(tf.float32, [None, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name='input_data')
    labels = tf.placeholder(tf.float32, [None, 2], name='labels')

    dropout_keep_prob = tf.placeholder_with_default(0.6, shape=(), name='dropout_keep_prob')
    
    weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)


    # current_pred, new_states = tf.nn.dynamic_rnn(cell, input_data, initial_state=rnn_tuple_state)

    # # new_states = tf.stack(new_states)

    # current_pred = tf.transpose(current_pred, [1, 0, 2])
    # current_pred = tf.gather(current_pred, int(current_pred.get_shape()[0] - 1))


    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=dropout_keep_prob)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, input_data, dtype=tf.float32)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.gather(outputs, int(outputs.get_shape()[0] - 1))

    prediction = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    # prediction = tf.nn.softmax(tf.matmul(current_pred, weights) + biases)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name='loss')

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

    Accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss




def train():
    def getTrainBatch():
        labels = []
        arr = np.zeros([BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE])
        for i in range(BATCH_SIZE):
            if (i % 2 == 0):
                num = randint(0, 12499)
                labels.append([1, 0])
            else:
                num = randint(12500, 24999)
                labels.append([0, 1])
            arr[i] = training_data_embedded[num, :, :]
        return arr, labels

    # Call implementation
    glove_array, glove_dict = load_glove_embeddings()

    training_data_text = load_data()
    training_data_embedded = embedd_data(training_data_text, glove_array, glove_dict)
    input_data, labels, dropout_keep_prob, optimizer, accuracy, loss = define_graph()

    # tensorboard
    tf.summary.scalar("training_accuracy", accuracy)
    tf.summary.scalar("loss", loss)
    summary_op = tf.summary.merge_all()

    # saver
    all_saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    logdir = "tensorboard/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in range(iterations):
        batch_data, batch_labels = getTrainBatch()
        sess.run(optimizer, {input_data: batch_data, labels: batch_labels,
                             dropout_keep_prob: 0.6})
        if (i % 50 == 0):
            loss_value, accuracy_value, summary = sess.run(
                [loss, accuracy, summary_op],
                {input_data: batch_data,
                 labels: batch_labels})
            writer.add_summary(summary, i)
            print("Iteration: ", i)
            print("loss", loss_value)
            print("acc", accuracy_value)
        if (i % SAVE_FREQ == 0 and i != 0):
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)
            save_path = all_saver.save(sess, checkpoints_dir +
                                       "/trained_model.ckpt",
                                       global_step=i)
            print("Saved model to %s" % save_path)
    sess.close()


def eval(data_path):
    glove_array, glove_dict = load_glove_embeddings()
    data_text = load_data(path=data_path)
    test_data = embedd_data(data_text, glove_array, glove_dict)

    num_samples = len(test_data)
    print("Loaded and preprocessed %s samples for evaluation" % num_samples)

    sess = tf.InteractiveSession()
    last_check = tf.train.latest_checkpoint('./checkpoints')
    saver = tf.train.import_meta_graph(last_check + ".meta")
    saver.restore(sess, last_check)
    graph = tf.get_default_graph()

    loss = graph.get_tensor_by_name('loss:0')
    accuracy = graph.get_tensor_by_name('accuracy:0')

    input_data = graph.get_tensor_by_name('input_data:0')
    labels = graph.get_tensor_by_name('labels:0')

    num_batches = num_samples // BATCH_SIZE
    label_list = [[1, 0]] * (num_samples // 2)  # pos always first, neg always second
    label_list.extend([[0, 1]] * (num_samples // 2))
    assert (len(label_list) == num_samples)
    total_acc = 0
    for i in range(num_batches):
        sample_index = i * BATCH_SIZE
        batch = test_data[sample_index:sample_index + BATCH_SIZE]
        batch_labels = label_list[sample_index:sample_index + BATCH_SIZE]
        lossV, accuracyV = sess.run([loss, accuracy], {input_data: batch,
                                                       labels: batch_labels})
        total_acc += accuracyV
        print("Accuracy %s, Loss: %s" % (accuracyV, lossV))
    print('-' * 40)
    print("FINAL ACC:", total_acc / num_batches)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "test"])

    args = parser.parse_args()

    if (args.mode == "train"):
        print("Training Run")
        train()
    elif (args.mode == "eval"):
        print("Evaluation run")
        eval("./data/validate")
    elif (args.mode == "test"):
        print("Test run")
        eval("./data/test")
