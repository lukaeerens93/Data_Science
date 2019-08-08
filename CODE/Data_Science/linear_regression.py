import keras
import pandas
from dataset_loader import *
from dataset_preprocessing import *
from regression import *



# ======== Linear Regression Test Script ===========


# 1) load, describe and visualize dataset
filename = 'boston_house_prices.csv'
d = dataset(filename)
d.show_dataset(20)
d.describe_data(stats='yes', corr='yes', skew='yes')
d.visualize_data(spread='histogram', correlation='scatter plot')
# Thought: What kind of story do the numbers tell, why might certain numbers be that way?


# 2) Preprocess dataset to better expose model to features of interest
values = d.data.values
prep = preprocess(values)
x,y = prep.seperate_input_output()
norm_x = prep.normalize(x,verbose=True)
# Thought: What features are most important, why might this be or not be the case?


# 3) Apply the model, with parameters of choice



# Thought: What considerations went into the choice of model and 

