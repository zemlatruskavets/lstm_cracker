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




#####################################################################################
#####################################################################################
#####################################################################################
######                                                                          #####
######                         LSTM PASSWORD GENERATOR                          #####
######                                                                          #####
#####################################################################################
#####################################################################################
#####################################################################################





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
import uuid
import yaml







class LSTM_network():

    def __init__():
        # variables


    def data_load():
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



    def parse_data():
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



    def model_construction():
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
        bool
            True if successful, False otherwise.

        """



    def model_training():
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
        bool
            True if successful, False otherwise.

        """



    def sequence_probability():
        """
        Calculate the probability of a given sequence.


        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.

        Returns
        -------
        float
            The probability of the sequence.

        """



    def guess():
        """
        Calculate the probability of a given sequence.


        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.

        Returns
        -------
        float
            The probability of the sequence.

        """



