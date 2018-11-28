import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split #for sklearn 0.20, grab from sklearn.model_selection
import keras
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
import functools

def disjunction(*values):
    return functools.reduce(np.logical_or, values)

#process csv file that we will use for project
dataframe = pd.read_csv('~/Downloads/clinvar_conflicting.csv')
row, _ = dataframe.shape

#fix chromosome (col 0)
dataframe.loc[:,'CHROM'].replace('X', 23, inplace=True)
dataframe.loc[:,'CHROM'].replace('MT', 24, inplace=True)
dataframe.loc[:,'CHROM'] = pd.to_numeric(dataframe.loc[:,'CHROM'])

#fix variant type (col 13)
dataframe.loc[:,'CLNVC'].replace('single_nucleotide_variant', 1, inplace=True)
dataframe.loc[:,'CLNVC'].replace('Duplication', 2, inplace=True)
dataframe.loc[:,'CLNVC'].replace('Deletion', 3, inplace=True)
dataframe.loc[:,'CLNVC'].replace('Indel', 4, inplace=True)
dataframe.loc[:,'CLNVC'].replace('Inversion', 5, inplace=True)
dataframe.loc[:,'CLNVC'].replace('Insertion', 6, inplace=True)
dataframe.loc[:,'CLNVC'].replace('Microsatellite', 7, inplace=True)
dataframe.loc[:,'CLNVC'] = pd.to_numeric(dataframe.loc[:,'CLNVC'])

#fix impact (col 20)
dataframe.loc[:,'IMPACT'].replace('MODERATE', 1, inplace=True)
dataframe.loc[:,'IMPACT'].replace('LOW', 2, inplace=True)
dataframe.loc[:,'IMPACT'].replace('MODIFIER', 3, inplace=True)
dataframe.loc[:,'IMPACT'].replace('HIGH', 4, inplace=True)
dataframe.loc[:,'IMPACT'] = pd.to_numeric(dataframe.loc[:,'IMPACT'])

#fix bam_edit (col 34)
dataframe.loc[:,'BAM_EDIT'].replace('OK', 0, inplace=True)
dataframe.loc[:,'BAM_EDIT'].replace('FAILED', 1, inplace=True)
dataframe.loc[:,'BAM_EDIT'] = pd.to_numeric(dataframe.loc[:,'BAM_EDIT'])

#fix sift (col 35)
dataframe.loc[:,'SIFT'].replace('deleterious', 1, inplace=True)
dataframe.loc[:,'SIFT'].replace('tolerated', 2, inplace=True)
dataframe.loc[:,'SIFT'].replace('deleterious_low_confidence', 3, inplace=True)
dataframe.loc[:,'SIFT'].replace('tolerated_low_confidence', 4, inplace=True)
dataframe.loc[:,'SIFT'] = pd.to_numeric(dataframe.loc[:,'SIFT'])

#fix polyphen (col 36)
dataframe.loc[:,'PolyPhen'].replace('benign', 1, inplace=True)
dataframe.loc[:,'PolyPhen'].replace('probably_damaging', 2, inplace=True)
dataframe.loc[:,'PolyPhen'].replace('possibly_damaging', 3, inplace=True)
dataframe.loc[:,'PolyPhen'].replace('unknown', 4, inplace=True)
dataframe.loc[:,'PolyPhen'] = pd.to_numeric(dataframe.loc[:,'PolyPhen'])


#extract start positions for CDNA, CDS, and protein (cols 25-27)
dataframe.loc[:,'cDNA_pos_start'] = dataframe.loc[:,'cDNA_position'].str.split('-').str.get(0)
dataframe.loc[:,'CDS_pos_start'] = dataframe.loc[:,'CDS_position'].str.split('-').str.get(0)
dataframe.loc[:,'Protein_pos_start'] = dataframe.loc[:,'Protein_position'].str.split('-').str.get(0)

#replace ? with null values in start position columns
dataframe.loc[:,'cDNA_pos_start'].replace('?',np.NaN,inplace=True)
dataframe.loc[:,'CDS_pos_start'].replace('?',np.NaN,inplace=True)
dataframe.loc[:,'Protein_pos_start'].replace('?',np.NaN,inplace=True)

#convert start position entries into floats
dataframe.loc[:,'cDNA_pos_start'] = pd.to_numeric(dataframe.loc[:,'cDNA_pos_start'])
dataframe.loc[:,'CDS_pos_start'] = pd.to_numeric(dataframe.loc[:,'CDS_pos_start'])
dataframe.loc[:,'Protein_pos_start'] = pd.to_numeric(dataframe.loc[:,'Protein_pos_start'])



#extract end positions for CDNA, CDS, and protein (cols 28-30)
dataframe.loc[:,'cDNA_pos_end'] = dataframe.loc[:,'cDNA_position'].str.split('-').str.get(1)
dataframe.loc[:,'CDS_pos_end'] = dataframe.loc[:,'CDS_position'].str.split('-').str.get(1)
dataframe.loc[:,'Protein_pos_end'] = dataframe.loc[:,'Protein_position'].str.split('-').str.get(1)

#replace ? with null values in end position columns
dataframe.loc[:,'cDNA_pos_end'].replace('?',np.NaN,inplace=True)
dataframe.loc[:,'CDS_pos_end'].replace('?',np.NaN,inplace=True)
dataframe.loc[:,'Protein_pos_end'].replace('?',np.NaN,inplace=True)

#convert start position entries into floats
dataframe.loc[:,'cDNA_pos_end'] = pd.to_numeric(dataframe.loc[:,'cDNA_pos_end'])
dataframe.loc[:,'CDS_pos_end'] = pd.to_numeric(dataframe.loc[:,'CDS_pos_end'])
dataframe.loc[:,'Protein_pos_end'] = pd.to_numeric(dataframe.loc[:,'Protein_pos_end'])


#delete columns containing miscellaneous or redundantinformation 
dataframe.drop(['CLNDISDB','CLNDISDBINCL','CLNDN','CLNDNINCL','CLNHGVS','CLNSIGINCL',
    'CLNVI','MC','ORIGIN','SSR','SYMBOL','Feature_type','Feature','BIOTYPE','Amino_acids',
    'Codons','DISTANCE','MOTIF_NAME','MOTIF_POS','HIGH_INF_POS','MOTIF_SCORE_CHANGE',
    'cDNA_position','CDS_position','Protein_position'],axis=1,inplace=True)

print(dataframe.dtypes)
    
info = [dataframe.iloc[i,:] for i in range(row)]
final_data = np.array(info)

#split dataset into x-value matrix and y-value array
fd_x = np.delete(final_data, 8, 1)
fd_y = final_data[:,8:9]

#split dataset into training and test data (former will have ~45k rows while latter will have ~20k rows)
train_data, test_data, train_label, test_label = train_test_split(fd_x, fd_y, train_size=0.7,
                                                    random_state=111, stratify=fd_y)

#Expand dimensions of input data
train_data = np.expand_dims(train_data, axis=2)
test_data = np.expand_dims(test_data, axis=2)

# convert class vectors to binary class matrices
train_label = keras.utils.to_categorical(train_label, num_classes=None)
test_label = keras.utils.to_categorical(test_label, num_classes=None)

#TODO: numeric (leave as values, DISCRETE)
'''
print(train_data[10:14,2]) #CHANGE TO VALUE (use ascii) REF
print(train_data[10:14,3]) #CHANGE TO VALUE (use ascii) ALT
print(train_data[10:14,8]) #CHANGE TO VALUE (use ascii) Allele
print(train_data[10:14,9]) #Consequence
print(train_data[10:14,11]) #CHANGE TO VALUE (take numerator and denominator - use '/' to split) EXON
print(train_data[10:14,12]) #CHANGE TO VALUE (take numerator and denominator - use '/' to split) INTRON
'''
print(train_data[10:14,24]) #CHANGE TO VALUE (simple change from string to int) cDNA_position
print(train_data[10:14,25]) #CHANGE TO VALUE (simple change from string to int) CDS_position
print(train_data[10:14,26]) #CHANGE TO VALUE (simple change from string to int) Protein_position
print(train_data[10:14,27]) #CHANGE TO VALUE (simple change from string to int) cDNA_position
print(train_data[10:14,28]) #CHANGE TO VALUE (simple change from string to int) CDS_position
print(train_data[10:14,29]) #CHANGE TO VALUE (simple change from string to int) Protein_position

'''
#TODO: categorical (CHANGE INTO VALUES, CONTINUOUS) strand, blosum62, and chrom are pre-handled
print(train_data[20:24,13]) #single_nucleotide_variant, Duplication, Deletion, Indel, Inversion, Insertion, Microsatellite
print(train_data[20:24,19]) #missense_variant, synonymous_variant, 3_prime_UTR_variant, 5_prime_UTR_variant
print(train_data[20:24,20]) #MODERATE, LOW, MODIFIER, HIGH
print(train_data[20:24,34]) #OK, FAILED
print(train_data[20:24,35]) #deleterious, tolerated, deleterious_low_confidence, tolerated_low_confidence
print(train_data[20:24,36]) #benign, probably damaging, possibly_damaging, unknown
'''


'''
#Build and compile the CNN model
print('CNN TEST: 32 3x3 CONV -> 2x2 MAXPOOL -> softmax')
model = Sequential()
model.add(Conv1D(32, kernel_size=3,
    activation='relu', input_shape=(48,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#Test the CNN model and display the run statistics
start_time = time.time()
model.fit(train_data, train_label, batch_size=128,
          epochs=1, verbose=1, validation_data=(test_data, test_label))
score = model.evaluate(test_data, test_label, verbose=0)
end_time = time.time()
total_time = end_time - start_time
print('Training time:',total_time)
print('Test accuracy:', score[1])
'''