
import gc
import keras
import logging
import modin.pandas as pd
import numpy as np
import pandas
import psutil
from keras.preprocessing.sequence import pad_sequences

# define logger
log = logging.getLogger("lstm-cracker-{}".format(datetime.date.today()))





class DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras """
    
    def __init__(self, data, vocabulary_size, max_length, batch_size, shuffle):
        """ Load the parameters """
        self.data            = data
        self.vocabulary_size = vocabulary_size
        self.max_length      = max_length
        self.batch_size      = batch_size
        self.shuffle         = shuffle

        # determine some of the dataset properties
        self.index_array = self.data.index.values
        self.data_length = len(self.data)
        self.on_epoch_end()



    def __len__(self):
        """ Determines the number of batches per epoch """
        
        return int(np.ceil(self.data_length / self.batch_size))



    def __getitem__(self, index):
        """ Generate one batch of data """
        
        # Generate a list of indices for a given batch
        indices = self.index_array[index*self.batch_size : (index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indices)

        # do a garbage collect to avoid memory leaks
        gc.collect()

        # if memory consumption is getting high, log a warning
        if psutil.virtual_memory().percent > 90:
            log.info('Memory usage has reached %g%%' % psutil.virtual_memory().percent)


        return X, y



    def on_epoch_end(self):
        """ shuffles indices after each epoch """
        
        if self.shuffle == True:
            np.random.shuffle(self.index_array)




    def slide_window(self, password):

        # Pads passwords and slides windows
        x = []

        for index, element in enumerate(password):
            x_padded = pad_sequences([password[:index]],
                                     maxlen=self.max_length - 1,
                                     padding='pre')[0]  # pads at the beginning
            x.append(x_padded)


        return x



    def __data_generation(self, indices, training_column='Tokenized', output_column='Output'):
        """ Generates data containing batch_size samples """

        # get a subset of the data
        data_subset = self.data.loc[indices]

        # get the data and its labels
        X = data_subset[training_column].apply(self.slide_window)

        # change lists to arrays
        X = X.apply(lambda x: np.array(x))

        # stack the arrays
        X = np.vstack(X.values)

        # generate the one-hot encoding of the output column
        y = data_subset[output_column].apply(lambda x: np.eye(self.vocabulary_size)[x])

        # change lists to arrays
        y = y.apply(lambda x: np.array(x))

        # stack the arrays
        y = np.vstack(y.values)

        # remove from memory
        del data_subset
        gc.collect()


        return X, y
