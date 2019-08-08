import keras

class preprocess:
	def __init__(self, dataset_values):
		self.dataset_values = dataset_values

	def seperate_input_output(self):
		#		inputs				 output
		# [x1,x2,x3,x4,x5 ...]  	 [x(-1)]
		inputs = self.dataset_values[:, 0:-1]
		outputs = self.dataset_values[:, -1]
		return inputs, outputs

	'''
	Functions for re-expressing dataset numbers
	'''
	def normalize(self, Input, verbose):
		# Recales attributes to hold similar range of values (typically b/w 0 and 1)
		# 0 and 1 range is useful for algos that weigh inputs like NN and regression
		# 0 and 1 range also useful for distance measuring algos like KNN
		from sklearn.preprocess import MinMaxScaler
		normalized_data = MinMaxScaler(feature_range = (0,1))
		normalized_input = normalized_data(Input)
		if verbose == True: print (normalized_input)
		return normalized_input

	def standardize(self, Input, verbose):
		# Used to transform attributes with Gaussian distribution but diverse means and std
		# into Gaussian distributions all of which are 0 mean and have a std of 1
		# Used for algos that assume Gaussian distributions like LDA, logistic Regression, regression
		from sklearn.preprocess import StandardScaler
		standardized_data = StandardScaler().fit(Input)
		if verbose == True: print(standardized_data)
		return standardized_data

	def binarize(self, Input, threshold, verbose):
		# Transforms dataset numbers into 1s and 0s based on if they are above or below a threshold value
		# You might need to normalize data before in case the domain of some attributes is huge, and small for others
		# so that the binarizing threshold is applicable to all attributes.
		from sklearn.preprocess import Binarizer
		binarize = Binarizer(threshold = threshold).fit(Input).transform(Input)
		if verbose == True: print (binarize)
		return binarize
		

	'''
	Functions for selecting the actual features of interest
	'''
	def univariate_selection(self, Input, Output, verbose):
		# Tells you which attributes have strongest relationship with output variable
		#  
		from sklearn.feature_selection import SelectKBest
		from sklearn.feature_selection import chi2
		features = SelectKBest(score_func=chi2, k=4).fit(Input, Output)
		if verbose == True: 
			print ("Class Scores: " + str(features.scores_) + " Transformed: " + str(features.transform(Input)))
		return features.transform(Input)

	def pca(self, n_comp, Input, verbose):
		# Finds component axes that mazimize variance in the data
		from sklearn.decomposition import PCA
		pca = PCA(n_components = n_comp).fit(Input)
		if verbose == True: print (pca)
		return pca

	def recursive_feature_elimination(self, Input, Output):
		# Recursively removes attributes and builds model on what remains (removes attributes that contribute least to accuracy) 
		# Requires an estimator (regression or SVM) as a parameter in order to determine accuracy
		from sklearn.feature_selection import RFE
		print ("This is not a completed function")
		pass




