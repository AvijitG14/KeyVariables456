import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split #for sklearn 0.20, grab from sklearn.model_selection
import keras
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import functools

'''
    TO SEE WHAT THE COLUMNS IN fd_x and fd_y REPRESENT,
    TAKE A LOOK IN THE csv_read_test PYTHON FILE.
'''

def disjunction(*values):
    return functools.reduce(np.logical_or, values)

#process csv file that we will use for project
dataframe = pd.read_csv('~/Downloads/clinvar_conflicting.csv')
row, _ = dataframe.shape

#fix chromosome 
dataframe.loc[:,'CHROM'].replace('X',23, inplace=True)
dataframe.loc[:,'CHROM'].replace('MT',24, inplace=True)
dataframe.loc[:,'CHROM'] = pd.to_numeric(dataframe.loc[:,'CHROM'])

#GOAL: convert object column dtypes into string
#split('/')[row][position = 0, length = 1]
#print(dataframe.loc[:,'EXON'].str.split('/')[0][1])
#print(dataframe.dtypes)

#extract start positions for CDNA, CDS, and protein
dataframe.loc[:,'cDNA_pos_start'] = dataframe.loc[:,'cDNA_position'].str.split('-').str.get(0)
dataframe.loc[:,'CDS_pos_start'] = dataframe.loc[:,'CDS_position'].str.split('-').str.get(0)
dataframe.loc[:,'Protein_pos_start'] = dataframe.loc[:,'Protein_position'].str.split('-').str.get(0)

#replace ? with null values in start position columns
dataframe.loc[:,'cDNA_pos_start'].replace('?',np.NaN,inplace=True)
dataframe.loc[:,'CDS_pos_start'].replace('?',np.NaN,inplace=True)
dataframe.loc[:,'Protein_pos_start'].replace('?',np.NaN,inplace=True)

'''
#mark unspecified CLNDN (DIS_NAME) and create new column
optionA = dataframe.loc[:,'CLNDN'] == "not_specified"
optionB = dataframe.loc[:,'CLNDN'] == "not_specified|not_provided"
optionC = dataframe.loc[:,'CLNDN'] == "not_provided|not_specified"
optionD = dataframe.loc[:,'CLNDN'] == "not_provided"
dataframe.loc[disjunction(optionA,optionB,optionC,optionD), 'CLNDN_not_specified'] = 1
dataframe.loc[:,'CLNDN_not_specified'].fillna(0, inplace=True)
'''

info = [dataframe.iloc[i,:] for i in range(row)]
final_data = np.array(info)

#split dataset into x-value matrix and y-value array
fd_x = np.delete(final_data, 18, 1)
fd_y = final_data[:,18:19]

#split dataset into training and test data (former has roughly 45k rows while latter has roughly 20k rows)
train_data, test_data, train_label, test_label = train_test_split(fd_x, fd_y, train_size=0.7,
                                                    random_state=111, stratify=fd_y)

# convert class vectors to binary class matrices
train_label = keras.utils.to_categorical(train_label, num_classes=None)
test_label = keras.utils.to_categorical(test_label, num_classes=None)

'''
input_shape = (10,10,1)

#TEST CONVOLUTIONAL NEURAL NETWORK
print('CNN TEST: 32 3x3 CONV -> 2x2 MAXPOOL -> softmax')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
    activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

start_time = time.time()
model.fit(train_data, train_label, batch_size=128,
          epochs=1, verbose=1, validation_data=(test_data, test_label))
score = model.evaluate(test_data, test_label, verbose=0)
end_time = time.time()
total_time = end_time - start_time
print('Training time:',total_time)
print('Test accuracy:', score[1])
'''