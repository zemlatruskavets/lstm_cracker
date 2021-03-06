# -*- coding: utf-8 -*-
"""

    This code trains a bidirectional long short-term memory (LSTM) 
    network on a dataset consisting solely of cleartext passwords.
    The model learns the patterns prevelant in password generation.
    It calculates the probability of a given password, which enables
    the comparison of different candidates for a password. This code
    will ultimately be used in a downstream task that will generate
    a password rules list in an iterative fashion.


Usage
-----
    It is assumed that this code is executed from a SageMaker notebook.
    Both the local and distributed modes have been tested. Please see 
    the Jupyter notebook (passwords.ipynb) for details.


Notes
-----
    The dataset is assumed to a single column containing the cleartext
    passwords.

    The network parameters (e.g., number of hidden units, embedding
    layer, etc.) are defined in the configuration file (config.yml)
    and in arguments passed along during the invokation step.

    This is the basic flow of the code:

        1) read in data
            1) read variables from config file
            2) parse arguments related to 
            3) clean up data (duplicates, NaN, etc)
        2) get data characteristics
            1) determine number of characters
            2) determine/define longest sequence length
            3) determine "vocabulary" size
        3) generator
            1) tokenization
            2) sliding windows
            3) persist vocab and tokenizer objects in S3
        4) model construction
            1) define model
            2) define generator objects
            3) train model
            4) persist model artifacts
        5) password probability prediction
            1) use the sliding window code to split up a given password
            2) use the model to calculate the conditional probabilities 
            3) take the sum of the logarithms of these probabilities
            4) calculate the perplexity of the resulting sum
            5) take the exponential of the resulting value


To do
-----
    1) load the model if it already exists (`model_construction`)
    2) get Pipe mode to work to stream data in during training
    3) Add method to deploy model endpoint after training

"""





################################################################################

##                              IMPORT MODULES                                ##

################################################################################


# full imports
import argparse
import boto3
import dateutil.parser as dp
import gc
import json
import logging
import modin.pandas as pd
import multiprocessing
import numpy as np
import os
import pandas
import pickle
import psutil
import random
import s3fs
import shutil as sh
import sys
import tensorflow as tf
import uuid
import yaml


# partial imports
from datetime                     import date, datetime, time, timedelta
from concurrent.futures           import ProcessPoolExecutor
from concurrent.futures           import ThreadPoolExecutor
from functools                    import partial
from generator                    import DataGenerator
from keras.callbacks              import ModelCheckpoint, EarlyStopping
from keras.layers                 import Embedding, LSTM, Dense, Bidirectional
from keras                        import backend as K
from keras.models                 import Sequential, load_model
from keras.preprocessing.text     import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils                  import to_categorical
from pympler.asizeof              import asizeof
from sklearn.model_selection      import train_test_split
from statistics                   import median
from tqdm                         import tqdm


# open access to S3 bucket
s3     = s3fs.S3FileSystem(s3_additional_kwargs={'ServerSideEncryption': 'AES256'})
client = boto3.client('s3')


# Import variables from config file
with open("config.yml", 'r') as config:
    variables = yaml.load(config, Loader=yaml.FullLoader)

# define logger
log = logging.getLogger("lstm-cracker-{}".format(date.today()))




class LSTM_network():
    """
    Train a BLSTM network on a cleartext password dataset.

    """

    def __init__(self):

        # load variables from the config file
        self.model_name      = variables['model']['name']
        self.gpu_count       = variables['model']['gpu_count']
        self.bucket          = variables['S3']['bucket_name']
        self.folder          = variables['S3']['folder']
        self.tokenizer_name  = variables['S3']['tokenizer_name']
        self.training_params = variables['S3']['training_params']
        self.history_pkl     = variables['S3']['history_pkl']


        # parse the arguments 
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs',       type=int, default=10)
        parser.add_argument('--batch_size',   type=int, default=128)
        parser.add_argument('--hidden_units', type=int, default=100)
        parser.add_argument('--training',     type=str)
        args, _ = parser.parse_known_args()
        
        # store the arguments as variables
        self.epochs          = args.epochs
        self.batch_size      = args.batch_size
        self.hidden_units    = args.hidden_units
        self.training_path   = args.training
        self.output_location = '%s/%s/output' % (self.bucket, self.folder)
        


    def data_load(self):
        """
        Load and clean the dataset from a specified location in S3.


        Parameters
        ----------
        training_path : str
            The path to the password dataset in S3.

        Returns
        -------
        data
            The cleaned dataset containing all of the passwords.

        """


        # read the dataset from an S3 bucket and store it as a pandas dataframe
        self.data = pd.read_csv(self.training_path, usecols=[0])

        # drop the rows with NaN values 
        self.data = self.data.dropna()

        # get rid of duplicate rows
        self.data = self.data.drop_duplicates()
        
        # truncate dataset
        self.data = self.data.head(10000)
        



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
        self.unique_characters = list(set(''.join(self.data['Password'])))
        self.vocabulary_size   = len(self.unique_characters)
        self.max_length        = self.data['Password'].str.len().max()

        
        
        
        
        
    def tokenization(self):
        """
        Tokenize the characters in the passwords.


        Parameters
        ----------
        data : pd.DataFrame
            The dataframe containing the passwords.
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
        with s3.open('%s/%s' % (self.output_location, self.tokenizer_name), 'w') as f:
            f.write(json.dumps(self.tokenizer.to_json(), ensure_ascii=False))
            
        # save the index-to-character dictionary and self.vocabulary_size values
        with s3.open('%s/%s' % (self.output_location, self.training_params), 'wb') as f:
            pickle.dump([self.ix_to_character, self.vocabulary_size, self.max_length], f)
            
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
        vocabulary_size : int
            The number of unique characters in the dataset.
        max_length : int
            The length of the longest password.
        hidden_units : int
            The number of hidden units in the LSTM network.

        Outputs
        -------
        model : 
            The Keras model.

        """

        # handle model loading


        # build the model
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocabulary_size + 1,              # vocabulary size plus an extra element for <PAD> 
                                 output_dim=int(self.vocabulary_size ** (1./4)),  # size of embeddings; fourth root of cardinality
                                 input_length=self.max_length - 1))               # length of the padded sequences
        self.model.add(Bidirectional(LSTM(self.hidden_units)))                    # size of hidden layer; n_h ~= n_s / (2(n_i + n_o)) 
        self.model.add(Dense(self.vocabulary_size, activation='softmax'))         # output
        self.model.compile('rmsprop', 'categorical_crossentropy')

        log.info(self.model.summary())






    def model_training(self):
        """
        Train the model.

        The dataset of tokenized passwords is split, using a sliding window, into 
        sublists of sequences of each password. The sliding window step is handled
        by the generator defined in generator.py. This process is used to generate 
        additional data that allows the network to learn the expected character given 
        an input sequence. This is ultimately how the probability of a given password
        is calculated.


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
        folder : str
            The name of the folder in the above S3 bucket in which the results are stored.
        history_pkl : str
            The name of the pickle object to store in S3.
        model_name : str
            The name of the model to be store in S3.


        Returns
        -------
        history : obj
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
        log.info("these are the memory stats prior to training: ")
        log.info(psutil.virtual_memory())

        log.info("starting training of model")

        # define the generators for the training and test datasets
        training_generator = DataGenerator(training, **paramaters)
        test_generator     = DataGenerator(testing, **paramaters)
        log.info(psutil.virtual_memory())

        # callbacks during training
        save_checkpoint = ModelCheckpoint(filepath       = '%s.h5' % self.model_name, 
                                          monitor        = 'val_accuracy', 
                                          save_best_only = True)
        early_stopping  = EarlyStopping(monitor='loss', patience=5)

        # add support for multiple GPUs
        if self.gpu_count > 1:
            self.model = multi_gpu_model(self.model, gpus=self.gpu_count)

        # train the network
        self.history = self.model.fit_generator(generator           = training_generator,
                                                validation_data     = test_generator,
                                                epochs              = self.epochs, 
                                                steps_per_epoch     = (len(training) // self.batch_size),
                                                validation_steps    = (len(testing) // self.batch_size),
                                                callbacks           = [save_checkpoint, early_stopping],
                                                use_multiprocessing = True,
                                                workers             = multiprocessing.cpu_count(),
                                                max_queue_size      = multiprocessing.cpu_count()*2,
                                                verbose             = 1).history

        # save the history variable
        with s3.open('%s/%s' % (self.output_location, self.history_pkl), 'wb') as f:
            pickle.dump(self.history, f)
        
        # save the hdf5 model in an S3 bucket
        self.model.save('%s.h5' % self.model_name)
        with open('%s.h5' % self.model_name, "rb") as f:
            client.upload_fileobj(Fileobj=f, 
                                  Bucket=self.bucket, 
                                  Key='%s/output/%s.h5' % (self.folder, self.model_name))

        # save Keras model for Tensorflow Serving in /opt/ml/model/1
        sess = K.get_session()
        tf.saved_model.simple_save(
            sess,
            os.path.join(os.environ['SM_MODEL_DIR'], '1'),
            inputs={'inputs': self.model.input},
            outputs={t.name: t for t in self.model.outputs})


        log.info("finished training model")





    def password_probability(self, password):
        """
        Calculate the probability of a given password. This works by 
        determining the product of the individual probabilities of a 
        given character conditional to the appearance of the preceding
        characters.


        Parameters
        ----------
        password : str
            The password whose probability is to be calculated.
        model : 
            The Keras model.
        tokenizer : 
            The Keras tokenizer object.
        ix_to_character : dict
            The index-to-character dictionary.
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

        # determine the probabilities of the permutations of the characters
        probabilities = self.model.predict(x_test, verbose=0)

        # multiply all of the conditional probabilities together in the password
        password_probability = 0
        for index, probability in enumerate(probabilities):
            char_probability      = probability[y_test[index]]  # get the probability from the model
            password_probability += np.log(char_probability)    # use log to avoid roundoff errors

        # calculate the perplexity to account for varying password lengths
        password_length       = len(password)    
        password_probability /= -password_length
        password_probability  = np.exp(password_probability)  # recover the raw probability


        return password_probability





# run the program
def main():

    # instantiate the class
    l = LSTM_network()

    # load the data
    l.data_load()

    # get the dataset characteristics
    l.parse_data()    

    # tokenize the data
    l.tokenization()

    # initialize the model
    l.model_construction()

    # train the model
    l.model_training()



if __name__ == "__main__":
    main()


