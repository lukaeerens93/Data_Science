# Dataset loader 
import os
import csv

class Data:
	def __init__(self, filename, method):
		self.filename = filename
		self.method = method
		if (os.path.isfile('./'+str(filename)) == False):
			print ('No such file in current directory')
		else:
			if (method == 'python'): data = numpy_method()
			if (method == 'pandas'): data = pandas_method()
			if (method == 'numpy'):  data = numpy_method()
			if (method == 'sklearn'): data = sklearn_method()


	def pandas_method(self, headings):
		'''
		1) if headings list not empty (assumes it is correct), 
		2) read the csv file using pandas library
		'''
		from pandas import read_csv
		d = []
		if not headings:
			print ('no headings provided')
		else:
			d = read_csv(self.filename, names = headings)
		return d 


	def python_method(self, delimeter=',', quoting=csv.QUOTE_NONE):
		'''
		1) Open file, 2) launch csv file reader, 
		3) convert read file into a list, 4) and then a np array
		'''
		import numpy as np
		raw_data = open(self.filename, 'rb')
		reader = csv.reader(raw_data, delimeter=delimeter, quoting=quoting)
		d = np.array(list(reader)).astype(float)
		return d


	def numpy_method(self, delimeter=','):
		'''
		1) Open file, 2) launch numpy loadtxt file reader (specify ',' delimeter, 
		'''
		import numpy as np
		raw_data = open(self.filename, 'rb')
		d = np.loadtxt(raw_data, delimeter = delimeter)
		return d


	def describe_dataset(dataset):
		'''
		This is a pandas function that statistically describes the dataset in terms of:
		count (total occurence), mean, std, min val, median + 1st quartile, 3rd quartile, max val
		'''
		desc = 0
		if (self.method = 'pandas'):
			from pandas import set_option
			set_option('display.width', 100)
			set_option('precision', 3)
			desc = dataset.describe()
		return desc


	def print_shape(self, dataset):
		if (dataset.type == np.array): print (dataset.shape)
		else: print ('not a numpy array')


	def sklearn_method(self):
		import sklearn
		d = []
		return d


	def visualize(self, dataset):
		import matplotlib.pyplot as plt

		v = []
		return v





# ----- Scriptor -------

# Load the data the numpy way
D = Data('dataset.csv', 'numpy')

# print the size of the array and than visualize it
dim = D.print_shape(D)
v = D.visualize(D)
