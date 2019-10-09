# -*- coding: utf-8 -*-
"""

This module trains a bidirectional long short-term memory (LSTM) 
network on a dataset consisting solely of cleartext passwords.
The trained network is then used to predict the most likely
alterations and/or additions to a given sequence.

Example
-------
    To run the program, include the dataset containing the cleartext 
    passwords as the first argument. The code will handle the rest.

    $ python3 program.py

Notes
-----
    The dataset is assumed to contain no information other than the 
    cleartext passwords.

    The network parameters (e.g., number of hidden units, embedding
    layer, etc.) are defined in the configuration file (config.yml).

    This is the basic flow of the code:

    1) read in data
        1) clean up data (duplicates, NaN, etc)
    2) get data characteristics
        1) determine number of characters
        2) determine/define longest sequence length
    3) generator
        3.1) tokenization
        3.2) sliding windows
    4) training
    5) sequence
        5.1) for i in sequence, predict most likely candidates in each position
        5.2) calculate most likely shared candidates
        5.3) calculate probabilities of overall adjusted sequences

Attributes
----------
s3 : str
    This variable holds connection information and allows typical file-system 
    style operations to interact with files stored in an S3 bucket.

variables : dict
    This dictionary holds the configuration variables defined in config.yml.
    

"""





################################################################################

##                              IMPORT MODULES                                ##

################################################################################


from generator                    import DataGenerator
from dask.diagnostics             import ProgressBar
from dask.multiprocessing         import get
from datetime                     import datetime, time, timedelta
from concurrent.futures           import ProcessPoolExecutor
from concurrent.futures           import ThreadPoolExecutor
from functools                    import partial
from keras.callbacks              import ModelCheckpoint, EarlyStopping
from keras.layers                 import Embedding, LSTM, Dense
from keras.models                 import Sequential, load_model
from keras.preprocessing.text     import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.utils                  import to_categorical
from pympler.asizeof              import asizeof
from sklearn.model_selection      import train_test_split
from statistics                   import median
from tqdm                         import tqdm


import dask.dataframe as dd
import dateutil.parser as dp
import gc
import json
import pickle
import multiprocessing
import numpy  as np
import pandas as pd
import psutil
import random
import s3fs
import settings
import sys
import uuid
import yaml



# open access to S3 bucket
s3 = s3fs.S3FileSystem(s3_additional_kwargs={'ServerSideEncryption': 'AES256'})

# Import variables from config file
with open("config.yml", 'r') as config:
    variables = yaml.load(config, Loader=yaml.FullLoader)





class LSTM_network():
    """
    Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note
    ----
    Do not include the `self` parameter in the ``Parameters`` section.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    data_path : str
        Path to dataset.

    """

    def __init__(self, ):
        self.epochs          = variables['model']['epochs']
        self.batch_size      = variables['model']['batch_size']
        self.hidden_units    = variables['model']['hidden_units']
        self.model_name      = variables['model']['name']
        self.data_path       = variables['S3']['data_path']
        self.bucket          = variables['S3']['bucket_name']
        self.tokenizer_name  = variables['S3']['tokenizer_name']
        self.training_params = variables['S3']['training_params']
        self.history_pkl     = variables['S3']['history_pkl']




    def data_load(self, ):
        """
        Load the data.


        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """


        # read the dataset from the S3 bucket and store it as a dask dataframe
        with s3.open('%s/%s.parquet' % (self.bucket, self.data_path), 'rb') as f:
            self.data = dd.read_parquet(f)

        # drop the rows with NaN values 
        self.data = self.data.dropna()

        # get rid of duplicate logs
        self.data = self.data.drop_duplicates()





    def parse_data(self, ):
        """
        Parse the data.


        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """


        self.unique_characters = sorted(list(set(''.join(self.data['sequences']))))
        self.vocabulary_size   = len(self.unique_characters)




    def tokenization(self, )

        # get the sequence column as its own array
        sequences = self.data['sequences']

        # define the tokenizer 
        self.tokenizer = Tokenizer(num_words=None, oov_token='UNK', char_level=True)

        # generate the tokenized sequences      
        self.tokenizer.fit_on_texts(sequences)

        # generate the word-to-index dictionary 
        self.word_to_ix = self.tokenizer.word_index

        # generate the index-to-word dictionary too
        self.ix_to_word = {i: j for j, i in self.word_to_ix.items()}

        # persist the tokenizer
        with s3.open('%s/%s' % (self.bucket, self.tokenizer_name), 'w') as f:
            f.write(json.dumps(self.tokenizer.to_json(), ensure_ascii=False))

        # save the index-to-word dictionary and self.vocabulary_size values
        with s3.open('%s/%s' % (self.bucket, self.training_params), 'wb') as f:
            pickle.dump([self.ix_to_word, self.self.vocabulary_size], f)

        # this encodes the sequences
        tokens = self.tokenizer.texts_to_sequences(sequences)

        # save the tokenized sequences in a column of the dataframe
        self.data['tokenized'] = tokens

        # turn the tokenized column into a column of arrays (not lists)
        self.data['tokenized'] = self.data['tokenized'].apply(lambda x: np.array(x))

        # this gets rid of the <PAD> character
        self.data['outputs'] = self.data['tokenized'] - 1





    def model_construction(self, ):
        """
        Construct the model.


        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.

        Returns
        -------
        float
            True if successful, False otherwise.

        """

        # define the generator parameters
        paramaters = {'self.vocabulary_size': self.self.vocabulary_size,
                      'max_length':           self.max_length,
                      'batch_size':           self.batch_size,
                      'shuffle':              True}

        # build the model
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.self.vocabulary_size + 1,             # vocabulary size plus an extra element for <PAD> 
                                output_dim=int(self.self.vocabulary_size ** (1./4)),  # size of embeddings; fourth root of cardinality
                                input_length=self.max_length - 1))                    # length of the padded sequences
        self.model.add(Bidirectional(LSTM(50)))                                       # size of hidden layer; n_h ~= n_s / (2(n_i + n_o)) 
        self.model.add(Dense(self.self.vocabulary_size, activation='softmax'))        # output
        self.model.compile('rmsprop', 'categorical_crossentropy')

        logger.info(self.model.summary())






    def model_training(self, ):
        """
        Train the model.


        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.

        Returns
        -------
        None

        """

        # split the data into training and testing sets
        training, testing = train_test_split(self.data, test_size=0.1)

        # determine some of the dataset properties
        training_index  = training.index.to_numpy()
        testing_index   = testing.index.to_numpy()
        training_length = len(training_index)
        testing_length  = len(testing_index)

        logger.info('finished getting data properties')

        # check memory
        logger.info(psutil.virtual_memory())


        logger.info("starting training of model")

        # define the generators for the training and test datasets
        training_generator = DataGenerator(training, training_index, training_length, **paramaters)
        test_generator     = DataGenerator(testing, testing_index, testing_length, **paramaters)
        logger.info(psutil.virtual_memory())

        # callbacks during training
        save           = ModelCheckpoint('%s.h5' % self.model_name, monitor='val_acc', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_acc', patience=5)

        # train network
        self.history = self.model.fit_generator(generator=training_generator,
                                 validation_data=test_generator,
                                 epochs=self.epochs, 
                                 callbacks=[save, early_stopping],
                                 steps_per_epoch=(training_length // self.batch_size),
                                 validation_steps=(testing_length // self.batch_size),
                                 use_multiprocessing=True,
                                 workers=2,
                                 max_queue_size=2,
                                 verbose=1).history

        # save the history variable
        with s3.open('%s/%s.pkl' % (self.bucket, self.history_pkl), 'wb') as f:
            pickle.dump(self.history, f)
        
        # save the model in an S3 bucket
        self.model.save('%s.h5' % self.model_name)
        with open('%s.h5' % self.model_name, "rb") as f:
            client.upload_fileobj(Fileobj=f, 
                                  Bucket=self.bucket, 
                                  Key='%s.h5' % self.model_name)


        logger.info("finished training model")





    def sequence_probability(self, sequence):
        """
        Calculate the probability of a given sequence.


        Parameters
        ----------
        sequence
            The sequence whose probability is to be calculated.

        Returns
        -------
        float
            The probability of the sequence.

        """

        # tokenize the sentence
        token  = self.tokenizer.texts_to_sequences([sentence])[0]
        x_test = self.slide_window(token)
        x_test = np.array(x_test)
        y_test = token - 1

        # determine the probabilities of the permutations of the words
        probabilities = self.model.predict(x_test, verbose=0)

        # calculate the probability of the sentence
        sequence_probability = 0
        for index, probability in enumerate(probabilities):
            word                  = self.ix_to_word[y_test[index] + 1]  # the first element is <PAD>
            word_probability      = probability[y_test[index]]          # get the probability from the model
            sequence_probability += np.log(word_probability)            # use log to avoid roundoff errors

        # calculate the perplexity to account for varying sequence lengths
        sentence_length       = len(sentence)    
        sequence_probability /= -sentence_length
        sentence_probability  = np.exp(sequence_probability)  # recover the raw probability


        return sentence_probability





    def guess(self, sequence, alterations, additions, top):
        """
        Calculate the probability of a given sequence.


        Parameters
        ----------
        sequence
            The sequence to be altered.
        alterations
            The number of characters in the input sequence to alter.
        additions
            The number of characters to add to the sequence.
        top
            The number of most likely candidates to return.

        Returns
        -------
        dict
            Keys correspond to the number of alterations.
            Values correspond to the top N guesses.

        """



