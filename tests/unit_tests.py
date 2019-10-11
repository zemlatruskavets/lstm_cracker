import unittest

# import the functions of the program
from program import LSTM_network


# https://pandas.pydata.org/pandas-docs/version/0.22.0/api.html#testing-functions

class DataTest(unittest.TestCase):

	def data_format(self):
		""" Ensure that the dataframe has the right format """
		self.assertEqual(fun(3), 4)

		# check that the dataset is not empty
		# check that data jusut has two columns
		# check the datatypes in the columns


	def data_format(self):
		""" Ensure that the dataframe has the right format """
		self.assertEqual(fun(3), 4)

	def data_format(self):
		""" Ensure that the dataframe has the right format """
		self.assertEqual(fun(3), 4)

	def data_format(self):
		""" Ensure that the dataframe has the right format """
		self.assertEqual(fun(3), 4)





class ModelTest(unittest.TestCase):

	def training(self):
		""" Ensure that the model can train on a test dataset """
		self.assertEqual(fun(3), 4)

	def inference(self):
		""" Ensure that the model can calculate an inference """
		self.assertEqual(fun(3), 4)







class OutputTest(unittest.TestCase):

	def inference(self):
		""" Ensure that the model can calculate an inference """
		self.assertEqual(fun(3), 4)