import pandas as pd
import numpy as np
import functools

'''
    This program performs basic access of elements within
    the dataset of the CSV file.
'''

def disjunction(*values):
    return functools.reduce(np.logical_or, values)

#process csv file that we will use for project
dataframe = pd.read_csv('~/Downloads/clinvar_conflicting.csv')

#type dataframe.CHROM as dataframe.loc[:,'CHROM']

#fix chromosome 
dataframe.loc[:,'CHROM'].replace('X',23, inplace=True)
dataframe.loc[:,'CHROM'].replace('MT',24, inplace=True)
dataframe.loc[:,'CHROM'] = pd.to_numeric(dataframe.loc[:,'CHROM'])

#GOAL: convert object column dtypes into string
#split('/')[row][position = 0, length = 1]
print(dataframe.loc[:,'EXON'].str.split('/')[0][1])
print(dataframe.dtypes)

#extract start positions for CDNA, CDS, and protein
dataframe.loc[:,'cDNA_pos_start'] = dataframe.loc[:,'cDNA_position'].str.split('-').str.get(0)
dataframe.loc[:,'CDS_pos_start'] = dataframe.loc[:,'CDS_position'].str.split('-').str.get(0)
dataframe.loc[:,'Protein_pos_start'] = dataframe.loc[:,'Protein_position'].str.split('-').str.get(0)

#replace ? with null values in start position columns
dataframe.loc[:,'cDNA_pos_start'].replace('?',np.NaN,inplace=True)
dataframe.loc[:,'CDS_pos_start'].replace('?',np.NaN,inplace=True)
dataframe.loc[:,'Protein_pos_start'].replace('?',np.NaN,inplace=True)

#initialize start position columns as floats
dataframe.loc[:,'cDNA_pos_start'] = dataframe.loc[:,'cDNA_pos_start'].astype(float)
dataframe.loc[:,'CDS_pos_start'] = dataframe.loc[:,'CDS_position'].astype(float)
dataframe.loc[:,'Protein_pos_start'] = dataframe.loc[:,'Protein_position'].astype(float)

#mark unspecified CLNDN (DIS_NAME) and create new column
optionA = dataframe.loc[:,'CLNDN'] == "not_specified"
optionB = dataframe.loc[:,'CLNDN'] == "not_specified|not_provided"
optionC = dataframe.loc[:,'CLNDN'] == "not_provided|not_specified"
optionD = dataframe.loc[:,'CLNDN'] == "not_provided"
dataframe.loc[disjunction(optionA,optionB,optionC,optionD), 'CLNDN_not_specified'] = 1
dataframe.loc[:,'CLNDN_not_specified'].fillna(0, inplace=True)

#info = [dataframe.iloc[i,:] for i in range(dataframe.shape[0])]

#convert CSV dataframe into numpy array for easy data manipulation
#final_data = np.array(info)
#fd_x = np.delete(final_data, 18, 1)
#fd_y = final_data[:,18:19]

"""
names will not be a 1-to-1 match because of
the name length of some columns in the original dataset.

*****REFERENCE X-VALUE COLUMNS*****

NUMERIC FEATURES
    POS = fd_x[:,1]
    REF = fd_x[:,2]
    ALT = fd_x[:,3]
    AF_ESP = fd_x[:,4]
    AF_EXAC = fd_x[:,5]

    AF_TGP = fd_x[:,6]
    ALLELE = fd_x[:,18]
    EXON = fd_x[:,25]
    INTRON = fd_x[:,26]
    CDNA_POS = fd_x[:,27]

    CDS_POS = fd_x[:,28]
    PRO_POS = fd_x[:,29]
    LOF_TOOL = fd_x[:,41]
    CADD_PHRED = fd_x[:,42]
    CADD_RAW = fd_x[:,43]

CATEGORICAL FEATURES
    CHROM = fd_x[:,0] (NOW TURNED INTO VALUES FROM 1-24)
    VAR_TYPE = fd_x[:,13]
    CNSQ = fd_x[:,19]
    IMPACT = fd_x[:,20]
    STRAND = fd_x[:,33]
    BAM_EDIT = fd_x[:,34]
    SIFT = fd_x[:,35]
    POLYPHEN = fd_x[:,36]
    BLOSUM62 = fd_x[:,44]
    
MISCELLANEOUS
    TAG_VAL = fd_x[:,7]
    TAG_VAL_VAR = fd_x[:,8]
    DIS_NAME = fd_x[:,9]
    DIS_NAME_VAR = fd_x[:,10]
    HGVS = fd_x[:,11]
    CLNSIG = fd_x[:,12]
    CLNSRC = fd_x[:,14]
    MC = fd_x[:,15]
    ORGN = fd_x[:,16]
    SSR = fd_x[:,17]                     
    SYMBOL = fd_x[:,21]
    FEAT_TYPE = fd_x[:,22]
    FEAT = fd_x[:,23]
    BIOTYPE = fd_x[:,24]      
    AMINO = fd_x[:,30]
    CODONS = fd_x[:,31]
    DST = fd_x[:,32]
    MOTIF_NAME = fd_x[:,37]
    MOTIF_POS = fd_x[:,38]
    HI_INF_POS = fd_x[:,39]
    MOTIF_SCORE = fd_x[:,40]         

*****REFERENCE Y-VALUE COLUMN (IMPORTANT)*****
    CLASS = fd_y

"""