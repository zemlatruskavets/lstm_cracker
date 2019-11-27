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
    The dataset is assumed to contain two columns: the usernames and the 
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
    

To do
-----
1) Put the parts saving data to S3 into a single function and invoke that
2) Think through the guessing code.

"""





################################################################################

##                              IMPORT MODULES                                ##

################################################################################


import argparse
import dateutil.parser as dp
import gc
import json
import multiprocessing
import numpy  as np
import pandas as pd
import pickle
import psutil
import random
import s3fs
import sys
import uuid
import yaml



from datetime                     import datetime, time, timedelta
from concurrent.futures           import ProcessPoolExecutor
from concurrent.futures           import ThreadPoolExecutor
from functools                    import partial
from generator                    import DataGenerator
from keras.callbacks              import ModelCheckpoint, EarlyStopping
from keras.layers                 import Embedding, LSTM, Dense, Bidirectional
from keras.models                 import Sequential, load_model
from keras.preprocessing.text     import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.utils                  import to_categorical
from pympler.asizeof              import asizeof
from sklearn.model_selection      import train_test_split
from statistics                   import median
from tqdm                         import tqdm



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

    def __init__(self):

        # load variables from the config file
        self.epochs          = variables['model']['epochs']
        self.batch_size      = variables['model']['batch_size']
        self.hidden_units    = variables['model']['hidden_units']
        self.gpu_count       = variables['model']['gpu_count']
        self.model_name      = variables['model']['name']
        self.data_path       = variables['S3']['data_path']
        self.bucket          = variables['S3']['bucket_name']
        self.tokenizer_name  = variables['S3']['tokenizer_name']
        self.training_params = variables['S3']['training_params']
        self.history_pkl     = variables['S3']['history_pkl']
        self.data_path       = variables['data']['path']


        parser = argparse.ArgumentParser()

        parser.add_argument('--epochs', type=int, default=10)
        
        args, _ = parser.parse_known_args()
        
        self.epochs     = args.epochs
        


    def data_load(self, data_location):
        """
        Load the data from some remote location.

        Parameters
        ----------
        data_location : str
            The path to the password dataset.

        Returns
        -------
        data
            The cleaned dataset containing all of the passwords.

        """


        # read the dataset from the S3 bucket and store it as a dask dataframe
        # with s3.open('%s/%s.parquet' % (self.bucket, self.data_path), 'rb') as f:
        #     self.data = dd.read_parquet(f)

        self.data = pd.read_csv(data_location)

        # drop the rows with NaN values 
        self.data = self.data.dropna()

        # get rid of duplicate rows
        self.data = self.data.drop_duplicates()

        # just keep some of the data
        self.data = self.data.head(1000)




    def parse_data(self):
        """
        Parse the data and determine some dataset properties.


        Parameters
        ----------
        data
            The cleaned dataset containing all of the passwords.

        Returns
        -------
        data_length : int
            The number of passwords in the dataset.
        unique_characters : int
            A sorted list of the unique characters in the dataset.
        vocabulary_size : int
            The number of unique characters in the dataset.
        max_length : int
            The length of the longest password in the dataset.

        """

        self.data_length       = len(self.data)
        self.unique_characters = sorted(list(set(''.join(self.data['Password']))))
        self.vocabulary_size   = len(self.unique_characters)
        self.max_length        = self.data['Password'].str.len().max()






    def tokenization(self):
        """
        Parse the data and determine some dataset properties.


        Parameters
        ----------
        data : pd.DataFrame
            The dataset containing the passwords.
        vocabulary_size : int
            The number of unique characters in the dataset.
        max_length : int
            The length of the longest password.
        bucket : str
            The name of the S3 bucket in which the results are stored.
        training_params : str
            The name of the pickle object to store in S3.
        tokenizer_name : str
            The name of the tokenizer object to be store in S3.


        Returns
        -------
        tokenizer : 
            The Keras tokenizer object.
        character_to_ix : 
            The character-to-index dictionary.
        ix_to_character : 
            The index-to-character dictionary.
        data : pd.DataFrame
            The dataset, including the tokenized passwords.

        """

        # get the password column as its own array
        passwords = self.data['Password']

        # define the tokenizer 
        self.tokenizer = Tokenizer(num_words=None, oov_token='UNK', char_level=True)

        # generate the tokenized passwords      
        self.tokenizer.fit_on_texts(passwords)

        # generate the character-to-index dictionary 
        self.character_to_ix = self.tokenizer.word_index

        # generate the index-to-character dictionary too
        self.ix_to_character = {i: j for j, i in self.character_to_ix.items()}

        # persist the tokenizer
        # with s3.open('%s/%s' % (self.bucket, self.tokenizer_name), 'w') as f:
        #     f.write(json.dumps(self.tokenizer.to_json(), ensure_ascii=False))

        # save the index-to-character dictionary and self.vocabulary_size values
        # with s3.open('%s/%s' % (self.bucket, self.training_params), 'wb') as f:
        #     pickle.dump([self.ix_to_character, self.vocabulary_size, self.max_length], f)

        # this encodes the passwords
        tokens = self.tokenizer.texts_to_sequences(passwords)

        # save the tokenized passwords in a column of the dataframe
        self.data['Tokenized'] = tokens

        # turn the tokenized column into a column of arrays (not lists)
        self.data['Tokenized'] = self.data['Tokenized'].apply(lambda x: np.array(x))

        # this gets rid of the <PAD> character
        self.data['Output'] = self.data['Tokenized'] - 1





    def model_construction(self):
        """
        Construct the model.


        Parameters
        ----------
        vocabulary_size
            The number of unique characters in the dataset.
        max_length
            The length of the longest password.

        Outputs
        -------
        model : 
            The Keras model.

        """


        # build the model
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocabulary_size + 1,             # vocabulary size plus an extra element for <PAD> 
                                output_dim=int(self.vocabulary_size ** (1./4)),  # size of embeddings; fourth root of cardinality
                                input_length=self.max_length - 1))               # length of the padded sequences
        self.model.add(Bidirectional(LSTM(50)))                                  # size of hidden layer; n_h ~= n_s / (2(n_i + n_o)) 
        self.model.add(Dense(self.vocabulary_size, activation='softmax'))        # output
        self.model.compile('rmsprop', 'categorical_crossentropy')

        print(self.model.summary())






    def model_training(self):
        """
        Train the model.


        Parameters
        ----------
        data : pd.DataFrame
            The dataset containing the passwords.
        vocabulary_size : int
            The number of unique characters in the dataset.
        max_length : int
            The length of the longest password.
        batch_size : int
            The number of samples to train during a single iteration.
        epoch_size : int
            The number of steps to train the model.
        model : 
            The Keras model created in model_construction.
        bucket : str
            The name of the S3 bucket in which the results are stored.
        history_pkl : str
            The name of the pickle object to store in S3.
        model_name : str
            The name of the model to be store in S3.


        Returns
        -------
        history : 
            The Keras history object.


        """

        # define the generator parameters
        paramaters = {'vocabulary_size': self.vocabulary_size,
                      'max_length':      self.max_length,
                      'batch_size':      self.batch_size,
                      'shuffle':         True}

        # split the data into training and testing sets
        training, testing = train_test_split(self.data, test_size=0.1)
 
        # check memory
        print("these are the memory stats prior to training: ")
        print(psutil.virtual_memory())

        print("starting training of model")

        # define the generators for the training and test datasets
        training_generator = DataGenerator(training, **paramaters)
        test_generator     = DataGenerator(testing, **paramaters)
        print(psutil.virtual_memory())

        # callbacks during training
        save_checkpoint = ModelCheckpoint('%s.h5' % self.model_name, monitor='val_acc', save_best_only=True)
        early_stopping  = EarlyStopping(monitor='loss', patience=5)

        # add support for multiple GPUs
        if self.gpu_count > 1:
            self.model = multi_gpu_model(self.model, gpus=self.gpu_count)

        # train the network
        self.history = self.model.fit_generator(generator=training_generator,
                                 validation_data=test_generator,
                                 epochs=self.epochs, 
                                 steps_per_epoch=(len(training) // self.batch_size),
                                 validation_steps=(len(testing) // self.batch_size),
                                 callbacks=[save_checkpoint, early_stopping],
                                 use_multiprocessing=True,
                                 workers=multiprocessing.cpu_count(),
                                 max_queue_size=multiprocessing.cpu_count()*2,
                                 verbose=1).history

        # save the history variable
        # with s3.open('%s/%s.pkl' % (self.bucket, self.history_pkl), 'wb') as f:
        #     pickle.dump(self.history, f)
        
        # save the model in an S3 bucket
        self.model.save('%s.h5' % self.model_name)
        # with open('%s.h5' % self.model_name, "rb") as f:
        #     client.upload_fileobj(Fileobj=f, 
        #                           Bucket=self.bucket, 
        #                           Key='%s.h5' % self.model_name)


        print("finished training model")





    def password_probability(self, password):
        """
        Calculate the probability of a given password.


        Parameters
        ----------
        password : str
            The password whose probability is to be calculated.
        model : 
            The Keras model.
        tokenizer : 
            The Keras tokenizer object.
        ix_to_character : 
            The c-to-character dictionary.
        data : pd.DataFrame
            The dataset, including the tokenized passwords.

        Returns
        -------
        float
            The probability of the password.

        """

        # tokenize the password
        token  = self.tokenizer.texts_to_sequences([password])[0]
        x_test = DataGenerator.slide_window(token)
        x_test = np.array(x_test)
        y_test = token - 1

        # determine the probabilities of the permutations of the words
        probabilities = self.model.predict(x_test, verbose=0)

        # calculate the probability of the password
        password_probability = 0
        for index, probability in enumerate(probabilities):
            word                  = self.ix_to_character[y_test[index] + 1]  # the first element is <PAD>
            word_probability      = probability[y_test[index]]               # get the probability from the model
            password_probability += np.log(word_probability)                 # use log to avoid roundoff errors

        # calculate the perplexity to account for varying password lengths
        password_length       = len(password)    
        password_probability /= -password_length
        password_probability  = np.exp(password_probability)  # recover the raw probability


        return password_probability





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



# run the program
def main():

    # # instantiate the class
    # l = LSTM_network()

    # # load the data
    # l.data_load('../data/Users.csv')

    # # get the dataset characteristics
    # l.parse_data()    

    # # tokenize the data
    # l.tokenization()

    # # initialize the model
    # l.model_construction()

    # # train the model
    # l.model_training()
    print('hello')



if __name__ == "__main__":
    main()


