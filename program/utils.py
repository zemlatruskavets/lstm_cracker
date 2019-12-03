# -*- coding: utf-8 -*-


import json
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer



def tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.
    # Arguments
        json_string: JSON string encoding a tokenizer configuration.
    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config           = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs  = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs   = word_docs
    tokenizer.index_docs  = index_docs
    tokenizer.word_index  = word_index
    tokenizer.index_word  = index_word

    return tokenizer


# def csv_to_tfrecords(csv_file, tfrecords_name):
    
#     read the csv values into a variable
#     csv = pandas.read_csv(csv_file).values
#     with tf.python_io.TFRecordWriter(tfrecords_name) as writer:
#         for row in csv:
#             example = tf.train.Example()
#             example.features.feature["row"].bytes_list.value.extend(features)
#             writer.write(example.SerializeToString())







def create_tf_example_row(input_row):

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'Password': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_row[0].encode('utf-8')]))
    }))
    return tf_example



def csv_to_tfrecords(csv_file, tfrecords_name):
    csv = pd.read_csv(csv_file).values
    with open(tf.python_io.TFRecordWriter(tfrecords_name)) as writer:
      for row in csv:
         example = create_tf_example(row)
         writer.write(example.SerializeToString())










