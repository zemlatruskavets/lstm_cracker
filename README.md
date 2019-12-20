    ██████╗ ██╗██╗     ███████╗████████╗███╗   ███╗     ██████╗██████╗  █████╗  ██████╗██╗  ██╗███████╗██████╗ 
    ██╔══██╗██║██║     ██╔════╝╚══██╔══╝████╗ ████║    ██╔════╝██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝██╔══██╗
    ██████╔╝██║██║     ███████╗   ██║   ██╔████╔██║    ██║     ██████╔╝███████║██║     █████╔╝ █████╗  ██████╔╝
    ██╔══██╗██║██║     ╚════██║   ██║   ██║╚██╔╝██║    ██║     ██╔══██╗██╔══██║██║     ██╔═██╗ ██╔══╝  ██╔══██╗
    ██████╔╝██║███████╗███████║   ██║   ██║ ╚═╝ ██║    ╚██████╗██║  ██║██║  ██║╚██████╗██║  ██╗███████╗██║  ██║
    ╚═════╝ ╚═╝╚══════╝╚══════╝   ╚═╝   ╚═╝     ╚═╝     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
----                                                                                                      

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
