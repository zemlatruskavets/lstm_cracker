# -*- coding: utf-8 -*-
​import json
import os
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
​
​
​
def tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.
​
    # Arguments
        json_string: JSON string encoding a tokenizer configuration.
    # Returns
        A Keras Tokenizer instance
    """
​
    tokenizer_config = json.loads(json_string)
    config           = tokenizer_config.get('config')
​
    word_counts = json.loads(config.pop('word_counts'))
    word_docs   = json.loads(config.pop('word_docs'))
    index_docs  = json.loads(config.pop('index_docs'))
    
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))
​
    tokenizer             = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs   = word_docs
    tokenizer.index_docs  = index_docs
    tokenizer.word_index  = word_index
    tokenizer.index_word  = index_word
​
​
    return tokenizer
​
​
​
​
​
def create_tf_example_row(input_row):
​
    # convert to string
    password = str(input_row[0])
​
    # create tf example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'Password': tf.train.Feature(bytes_list=tf.train.BytesList(value=[password.encode('utf-8')]))
    }))
    return tf_example
​
​
​
​
​
def batch_processing(csv, outfile, previous, file_number, finished, file_size=1e8):
​
    # this is the file name of the TFRecord
    file_name = '{}-{}.tf'.format(outfile, file_number)
    with tf.python_io.TFRecordWriter(file_name) as writer:
        for index, row in enumerate(csv[previous:], start=previous):
            example = create_tf_example_row(row)
            writer.write(example.SerializeToString())
​
            # if the file size is greater than the threshold, quit and return the values
            if os.path.getsize(file_name) > file_size:
                previous     = index
                file_number += 1
                return previous, file_number, finished
​
            # if the index is at the end of the file, set 'finished' to True
            else:
                if index == len(csv) - 1:
                    finished = True
                    return previous, file_number, finished
​
​
​
​
​
def csv_to_tfrecords(csv_file, outfile):
​
    # read in the csv (or txt) file
    csv = pd.read_csv(csv_file, usecols=[0]).values
​
    # set the initial variables for the conversion process
    previous    = 0
    file_number = 0
    finished    = False
​
    # convert the csv file to a TFRecord format in chunks
    while True:
        previous, file_number, finished = batch_processing(csv, outfile, previous, file_number, finished)
        if finished is True:
            break
​
​
​
​
def parser(record):
​
    # define the format
    passwords = {
      'Password': tf.FixedLenFeature([], tf.string)
    }
​
    # map the feature keys to tensors
    parsed = tf.parse_single_example(record, passwords)
​
    # ensure that the passwords are strings
    passwords = tf.cast(parsed['Password'], tf.string)    
​
​
    # return the passwords as a dictionary
    return {'Password': passwords}