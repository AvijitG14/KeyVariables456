import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split #for sklearn 0.20, grab from sklearn.model_selection

'''
    TO SEE WHAT THE COLUMNS IN fd_x and fd_y REPRESENT,
    TAKE A LOOK IN THE csv_read_test PYTHON FILE.
'''

#process csv file that we will use for project
dataframe = pd.read_csv('~/Downloads/clinvar_conflicting.csv')
row, _ = dataframe.shape
info = [dataframe.iloc[i,:] for i in range(row)]
final_data = np.array(info)

#split dataset into x-value matrix and y-value array
fd_x = np.delete(final_data, 18, 1)
fd_y = final_data[:,18:19]

#split dataset into training and test data (former has roughly 45k rows while latter has roughly 20k rows)
train_data, test_data, train_label, test_label = train_test_split(fd_x, fd_y, train_size=0.7,
                                                    random_state=111, stratify=fd_y)