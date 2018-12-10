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
from sklearn import svm
import functools


def DataframeAnd(*conditions):
    return functools.reduce(np.logical_and, conditions)

def ConsequenceRows(df, row_name):
    df.loc[:,row_name].replace('missense_variant',1,inplace=True)
    df.loc[:,row_name].replace('synonymous_variant',2,inplace=True)
    df.loc[:,row_name].replace('splice_acceptor_variant',3,inplace=True)
    df.loc[:,row_name].replace('splice_donor_variant',4,inplace=True)
    df.loc[:,row_name].replace('stop_gained',5,inplace=True)
    df.loc[:,row_name].replace('frameshift_variant',6,inplace=True)
    df.loc[:,row_name].replace('stop_lost',7,inplace=True)
    df.loc[:,row_name].replace('inframe_insertion',8,inplace=True)
    df.loc[:,row_name].replace('inframe_deletion',9,inplace=True)
    df.loc[:,row_name].replace('protein_altering_variant',10,inplace=True)
    df.loc[:,row_name].replace('splice_region_variant',11,inplace=True)
    df.loc[:,row_name].replace('start_retained_variant',12,inplace=True)
    df.loc[:,row_name].replace('stop_retained_variant',13,inplace=True)
    df.loc[:,row_name].replace('coding_sequence_variant',14,inplace=True)
    df.loc[:,row_name].replace('5_prime_UTR_variant',15,inplace=True)
    df.loc[:,row_name].replace('3_prime_UTR_variant',16,inplace=True)
    df.loc[:,row_name].replace('intron_variant',17,inplace=True)
    df.loc[:,row_name].replace('non_coding_transcript_variant',18,inplace=True)
    df.loc[:,row_name].replace('upstream_gene_variant',19,inplace=True)
    df.loc[:,row_name].replace('downstream_gene_variant',20,inplace=True)
    df.loc[:,row_name].replace('TF_binding_site_variant',21,inplace=True)
    df.loc[:,row_name].replace('intergenic_variant',22,inplace=True)
    df.loc[:,row_name].replace('start_lost',23,inplace=True)

    return df

def ConsequenceValues(df):
    #create two consequence columns to support a row having multiple entries
    df.loc[:,'Cons_one'] = df.loc[:,'Consequence'].str.split('&').str.get(0)
    df.loc[:,'Cons_two'] = df.loc[:,'Consequence'].str.split('&').str.get(1)
        
    #change all consequence types to numbers
    df = ConsequenceRows(df, 'Cons_one')
    df = ConsequenceRows(df, 'Cons_two')

    #convert consequence entries into floats
    df.loc[:,'Cons_one'] = pd.to_numeric(df.loc[:,'Cons_one'])
    df.loc[:,'Cons_two'] = pd.to_numeric(df.loc[:,'Cons_two'])
    
    return df

def CategoryColumnChanges(df):
    #fix chromosome (col 0)
    df.loc[:,'CHROM'].replace('X', 23, inplace=True)
    df.loc[:,'CHROM'].replace('MT', 24, inplace=True)
    df.loc[:,'CHROM'] = pd.to_numeric(df.loc[:,'CHROM'])
    
    #fix variant type (col 13)
    df.loc[:,'CLNVC'].replace('single_nucleotide_variant', 1, inplace=True)
    df.loc[:,'CLNVC'].replace('Duplication', 2, inplace=True)
    df.loc[:,'CLNVC'].replace('Deletion', 3, inplace=True)
    df.loc[:,'CLNVC'].replace('Indel', 4, inplace=True)
    df.loc[:,'CLNVC'].replace('Inversion', 5, inplace=True)
    df.loc[:,'CLNVC'].replace('Insertion', 6, inplace=True)
    df.loc[:,'CLNVC'].replace('Microsatellite', 7, inplace=True)
    df.loc[:,'CLNVC'] = pd.to_numeric(df.loc[:,'CLNVC'])
    
    #fix impact (col 20)
    df.loc[:,'IMPACT'].replace('MODERATE', 1, inplace=True)
    df.loc[:,'IMPACT'].replace('LOW', 2, inplace=True)
    df.loc[:,'IMPACT'].replace('MODIFIER', 3, inplace=True)
    df.loc[:,'IMPACT'].replace('HIGH', 4, inplace=True)
    df.loc[:,'IMPACT'] = pd.to_numeric(df.loc[:,'IMPACT'])
    
    #fix bam_edit (col 34)
    df.loc[:,'BAM_EDIT'].replace('OK', 0, inplace=True)
    df.loc[:,'BAM_EDIT'].replace('FAILED', 1, inplace=True)
    df.loc[:,'BAM_EDIT'] = pd.to_numeric(df.loc[:,'BAM_EDIT'])
    
    #fix sift (col 35)
    df.loc[:,'SIFT'].replace('deleterious', 1, inplace=True)
    df.loc[:,'SIFT'].replace('tolerated', 2, inplace=True)
    df.loc[:,'SIFT'].replace('deleterious_low_confidence', 3, inplace=True)
    df.loc[:,'SIFT'].replace('tolerated_low_confidence', 4, inplace=True)
    df.loc[:,'SIFT'] = pd.to_numeric(df.loc[:,'SIFT'])
    
    #fix polyphen (col 36)
    df.loc[:,'PolyPhen'].replace('benign', 1, inplace=True)
    df.loc[:,'PolyPhen'].replace('probably_damaging', 2, inplace=True)
    df.loc[:,'PolyPhen'].replace('possibly_damaging', 3, inplace=True)
    df.loc[:,'PolyPhen'].replace('unknown', 4, inplace=True)
    df.loc[:,'PolyPhen'] = pd.to_numeric(df.loc[:,'PolyPhen'])

    return df

def StartEndPositions(df):
    #extract start positions for CDNA, CDS, and protein (cols )
    df.loc[:,'cDNA_pos_start'] = df.loc[:,'cDNA_position'].str.split('-').str.get(0)
    df.loc[:,'CDS_pos_start'] = df.loc[:,'CDS_position'].str.split('-').str.get(0)
    df.loc[:,'Protein_pos_start'] = df.loc[:,'Protein_position'].str.split('-').str.get(0)
    
    #replace ? with null values in start position columns
    df.loc[:,'cDNA_pos_start'].replace('?',np.NaN,inplace=True)
    df.loc[:,'CDS_pos_start'].replace('?',np.NaN,inplace=True)
    df.loc[:,'Protein_pos_start'].replace('?',np.NaN,inplace=True)
    
    #convert start position entries into floats
    df.loc[:,'cDNA_pos_start'] = pd.to_numeric(df.loc[:,'cDNA_pos_start'])
    df.loc[:,'CDS_pos_start'] = pd.to_numeric(df.loc[:,'CDS_pos_start'])
    df.loc[:,'Protein_pos_start'] = pd.to_numeric(df.loc[:,'Protein_pos_start'])
    
    #extract end positions for CDNA, CDS, and protein (cols )
    df.loc[:,'cDNA_pos_end'] = df.loc[:,'cDNA_position'].str.split('-').str.get(1)
    df.loc[:,'CDS_pos_end'] = df.loc[:,'CDS_position'].str.split('-').str.get(1)
    df.loc[:,'Protein_pos_end'] = df.loc[:,'Protein_position'].str.split('-').str.get(1)
    
    #replace ? with null values in end position columns
    df.loc[:,'cDNA_pos_end'].replace('?',np.NaN,inplace=True)
    df.loc[:,'CDS_pos_end'].replace('?',np.NaN,inplace=True)
    df.loc[:,'Protein_pos_end'].replace('?',np.NaN,inplace=True)
    
    #convert start position entries into floats
    df.loc[:,'cDNA_pos_end'] = pd.to_numeric(df.loc[:,'cDNA_pos_end'])
    df.loc[:,'CDS_pos_end'] = pd.to_numeric(df.loc[:,'CDS_pos_end'])
    df.loc[:,'Protein_pos_end'] = pd.to_numeric(df.loc[:,'Protein_pos_end'])
    
    return df

def FindAlleleLengths(df):
    #extract lengths for REF, ALT, Allele (cols )
    df.loc[:,'REF_len'] = df.loc[:,'REF'].str.len()
    df.loc[:,'ALT_len'] = df.loc[:,'ALT'].str.len()
    df.loc[:,'Allele_len'] = df.loc[:,'Allele'].str.len()
    
    #convert data types  of columns to float
    df.loc[:,'REF_len'] = pd.to_numeric(df.loc[:,'REF_len'])
    df.loc[:,'ALT_len'] = pd.to_numeric(df.loc[:,'ALT_len'])
    df.loc[:,'Allele_len'] = pd.to_numeric(df.loc[:,'Allele_len'])
    
    return df

def ExonIntronProcessing(df):
    #detect exon/intron presence
    df.loc[:,'Exon_found'] = df.loc[:,'EXON'].notnull().astype(int)
    df.loc[:,'Intron_found'] = df.loc[:,'INTRON'].notnull().astype(int)
    
    #find exon position
    df.loc[:,'Exon_pos'] = df.loc[:,'EXON'].str.split('/').str.get(0)
    df.loc[:,'Exon_pos'] = pd.to_numeric(df.loc[:,'Exon_pos'])
    
    #find exon length
    df.loc[:,'Exon_length'] = df.loc[:,'EXON'].str.split('/').str.get(1)
    df.loc[:,'Exon_length'] = pd.to_numeric(df.loc[:,'Exon_length'])
    
    #turn null pos and length values into zero values
    exon_pos_empty = df.loc[:,'Exon_pos'].isnull()
    exon_length_empty = df.loc[:,'Exon_length'].isnull()
    exon_filter = DataframeAnd(exon_pos_empty, exon_length_empty)
    df.loc[exon_filter, 'Exon_pos'] = 0
    df.loc[exon_filter, 'Exon_length'] = 0
    
    return df

#process csv file that we will use for project
dataframe = pd.read_csv('~/Downloads/clinvar_conflicting.csv')
row, _ = dataframe.shape

#simplify various data columns
dataframe = CategoryColumnChanges(dataframe)
dataframe = StartEndPositions(dataframe)
dataframe = FindAlleleLengths(dataframe)
dataframe = ExonIntronProcessing(dataframe)
dataframe = ConsequenceValues(dataframe)

#delete columns containing miscellaneous or redundant information 
dataframe.drop(['CLNDISDB','CLNDISDBINCL','CLNDN','CLNDNINCL','CLNHGVS','CLNSIGINCL',
    'CLNVI','MC','ORIGIN','SSR','SYMBOL','Feature_type','Feature','BIOTYPE','Amino_acids',
    'Codons','DISTANCE','MOTIF_NAME','MOTIF_POS','HIGH_INF_POS','MOTIF_SCORE_CHANGE',
    'cDNA_position','CDS_position','Protein_position','REF','ALT','Allele','EXON','INTRON',
    'Consequence'],
    axis=1,inplace=True)

dataframe.fillna(0,inplace=True)

info = [dataframe.iloc[i,:] for i in range(row)]
final_data = np.array(info)

#split dataset into x-value matrix and y-value array
fd_x = np.delete(final_data, 6, 1)
fd_y = final_data[:,6:7]

#split dataset into training and test data (former will have ~45k rows while latter will have ~20k rows)
train_data, test_data, train_label, test_label = train_test_split(fd_x, fd_y, train_size=45000,
                                                    random_state=None, stratify=fd_y)

#Create the SVM model
print('Build SVM')
#model = svm.SVC(kernel='linear', C=1, gamma=1,verbose=True)
model = svm.LinearSVC(loss='hinge', tol=1e-5, max_iter=5000)

print('training SVM...') 
start_time = time.time()
model.fit(train_data, train_label.ravel())
print('finding the maximum margin')
model.decision_function(train_data)
end_time = time.time()
#model.get_params(deep=TRUE)

total_time = end_time - start_time
print('done')
print('Training time:',total_time)
print('Test accuracy', model.score(train_data, train_label))

predictions = model.predict(test_data)

#Expand dimensions of input data
train_data = np.expand_dims(train_data, axis=2)
test_data = np.expand_dims(test_data, axis=2)

# convert class vectors to binary class matrices
train_label = keras.utils.to_categorical(train_label, 2)
test_label = keras.utils.to_categorical(test_label, 2)


#Build and compile the CNN model
#current test accuracy without consq = 74.79%
#regardless of epoch size or convolution layers
print('CNN TEST: 8 2-length CONV -> softmax')
model = Sequential()
model.add(Conv1D(8, kernel_size=2,
    activation='sigmoid', input_shape=(train_data.shape[1],1)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#Test the CNN model and display the run statistics
start_time = time.time()
model.fit(train_data, train_label, batch_size=32,
          epochs=6, verbose=1, validation_data=(test_data, test_label))
score = model.evaluate(test_data, test_label, verbose=0)
end_time = time.time()
total_time = end_time - start_time
print('Training time:',total_time)
print('Test accuracy:', score[1])

for layer in model.layers:
    w = layer.get_weights()
    print(w)