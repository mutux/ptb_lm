import collections
import tensorflow as tf
import os


def _read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (x[-1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def _epoch_slicer(raw_data, batch_size, slice_size, name=None):
    # One batch for each step
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    num_data = tf.size(raw_data)
    num_batch = num_data // batch_size
    data = tf.reshape(raw_data[0: batch_size * num_batch], [batch_size, num_batch])
    num_slice = (num_batch - 1) // slice_size
    # A background QueueRunner thread will be started. Data will be queued as epoch_size share.
    i = tf.train.range_input_producer(num_slice, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * slice_size], [batch_size, (i + 1) * slice_size])
    x.set_shape([batch_size, slice_size])
    y = tf.strided_slice(data, [0, i * slice_size + 1], [batch_size, (i + 1) * slice_size + 1])
    y.set_shape([batch_size, slice_size])
    return x, y


def get_raw_data(data_path):
    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')
    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


class Input(object):
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.slice_size = slice_size = config.slice_size
        self.num_slice = ((len(data) // batch_size) - 1) // slice_size
        self.input_data, self.targets = _epoch_slicer(
            data, batch_size, slice_size, name=name)
