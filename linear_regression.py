import keras
import pandas
import dataset_loader as dl



# ======== Linear Regression Test Script ===========


# 1) load, describe and visualize dataset
filename = 'boston_house_prices.csv'
d = dl.dataset(filename)
d.dl.show_dataset(20)
d.dl.describe_data(stats='yes', corr='yes', skew='yes')
d.dl.visualize_data(spread='histogram', correlation='scatter plot')
# What kind of story do the numbers tell, why might certain numbers be that way. Think about it.

# 


