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

    $ python3 program.py <path_to_dataset>

Notes
-----
    The dataset is assumed to contain no information other than the 
    cleartext passwords.

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
    This variable holds connection information and allows typical file-system style
    commands to interact with files stored in an S3 bucket.

variables : dict
    This dictionary holds the configuration variables defined in config.yml
    

"""





#####################################################################################

##                                 IMPORT MODULES                                  ##

#####################################################################################


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

    def __init__():
        # variables


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



