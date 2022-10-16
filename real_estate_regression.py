# Import modules we'll need for this notebook
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import joblib

# read and load csv
real_estate_data = pd.read_csv(
    "/Users/hiro0x/Documents/microsoft/train-eval-regression/daily-bike-share.csv"
)
# print the first n lines of the data as read
real_estate_data.head(10)
# get the label column
label = real_estate_data[real_estate_data.columns[-1]]
# create a subplot for 2 subplots - 2 rows, 1 column
fig, ax = plt.subplots(2, 1, figsize=(9, 12))

# plot the histogram
ax[0].hist(label, bins=100)
ax[0].set_ylabel("Price Per Unit")

# ass lines for mean & median
ax[0].axvline(label.mean(), color="red", linestyle="dashed", linewidth=2)
ax[0].axvline(label.median(), color="blue", linestyle="dashed", linewidth=2)

# plot the boxplot
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel("Label")

# add a title to the figure
fig.suptitle("Label Distribution")

fig.show()
print(real_estate_data.head(15))
