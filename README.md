This module trains a bidirectional long short-term memory (LSTM) 
network on a dataset consisting of cleartext passwords.
The trained network is then used to calculate the probability of a given password. 
This is used for a downstream process that iteratively generates rules lists for password dumps.

Example
-------
To run the program, include the dataset containing the cleartext 
passwords as the first argument. The code will handle the rest.

    $ python3 program.py <path_to_dataset>

Notes
-----
The dataset is assumed to contain no information other than the 
cleartext passwords.

The network parameters (e.g., number of hidden units, embedding
layer, etc.) are defined in the configuration file (`config.yml`).

This is the basic flow of the code:

1) read in data
    1) clean up data (duplicates, NaN, etc)
2) get data characteristics
    1) determine number of characters
    2) determine/define longest sequence length
3) generator
    1) tokenization
    2) sliding windows
4) training
5) sequence
    1) for i in sequence, predict most likely candidates in each position
    2) calculate most likely shared candidates
    3) calculate probabilities of overall adjusted sequences

Attributes
----------
    s3 : str
        This variable holds connection information and allows typical file-system 
        style operations to interact with files stored in an S3 bucket.

    variables : dict
        This dictionary holds the configuration variables defined in config.yml
    
