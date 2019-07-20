import keras
import pandas
from dataset_loader import *



# ======== Linear Regression Test Script ===========


# 1) load, describe and visualize dataset
filename = 'boston_house_prices.csv'
d = dataset(filename)
d.show_dataset(20)
d.describe_data(stats='yes', corr='yes', skew='yes')
d.visualize_data(spread='histogram', correlation='scatter plot')
# What kind of story do the numbers tell, why might certain numbers be that way. Think about it.

# 


