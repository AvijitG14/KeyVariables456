import pandas as pd
import numpy as np

'''
    This program performs basic access of elements within
    the dataset of the CSV file.
'''

#process csv file that we will use for project
dataframe = pd.read_csv('~/Downloads/clinvar_conflicting.csv')

dataframe.loc[:,'CHROM'].replace('X',23, inplace=True)
dataframe.loc[:,'CHROM'].replace('MT',24, inplace=True)

row, _ = dataframe.shape
info = [dataframe.iloc[i,:] for i in range(row)]

#convert CSV dataframe into numpy array for easy data manipulation
final_data = np.array(info)
fd_x = np.delete(final_data, 18, 1)
fd_y = final_data[:,18:19]

#VALUES EXON AND INTRON COLUMNS CAN BE REDUCED TO 0/1
#IF EXON IS NOT NULL, MAKE IT 1 AND MAKE INTRON 0
#IF INTRON IS NOT NULL, MAKE IT 1 AND MAKE EXON 0

"""
names will not be a 1-to-1 match because of
the name length of some columns in the original dataset.

REFERENCE COLUMN NAMES - X-VALUES
    CHROM = fd_x[:,0]         POS = fd_x[:,1]            REF = fd_x[:,2]
    ALT = fd_x[:,3]           AF_ESP = fd_x[:,4]         AF_EXAC = fd_x[:,5]
    AF_TGP = fd_x[:,6]        TAG_VAL = fd_x[:,7]        TAG_VAL_VAR = fd_x[:,8]
    DIS_NAME = fd_x[:,9]      DIS_NAME_VAR = fd_x[:,10]  HGVS = fd_x[:,11]
    CLNSIG = fd_x[:,12]       VAR_TYPE = final_data[:,13]      CLNSRC = final_data[:,14]
    MC = fd_x[:,15]           ORGN = fd_x[:,16]          SSR = fd_x[:,17]
    ALLELE = fd_x[:,18]       CNSQ = fd_x[:,19]          IMPACT = fd_x[:,20]
    SYMBOL = fd_x[:,21]       FEAT_TYPE = fd_x[:,22]     FEAT = fd_x[:,23]
    BIOTYPE = fd_x[:,24]      EXON = fd_x[:,25]          INTRON = fd_x[:,26]
    CDNA_POS = fd_x[:,27]     CDS_POS = fd_x[:,28]       PRO_POS = fd_x[:,29]
    AMINO = fd_x[:,30]        CODONS = fd_x[:,31]        DST = fd_x[:,32]
    STRAND = fd_x[:,33]       BAM_EDIT = fd_x[:,34]      SIFT = fd_x[:,35]
    POLYPHEN = fd_x[:,36]     MOTIF_NAME = fd_x[:,37]    MOTIF_POS = fd_x[:,38]
    HI_INF_POS = fd_x[:,39]   MOTIF_SCORE = fd_x[:,40]   LOF_TOOL = fd_x[:,41]
    CADD_PHRED = fd_x[:,42]   CADD_RAW = fd_x[:,43]      BLOSUM62 = fd_x[:,44]


REFERENCE COLUMN NAMES - Y-VALUE
    CLASS = fd_y <-THIS IS THE Y-VALUE, IMPORTANT

"""

#perform dataset element access as you would with a 2-D array in Python
first_row_tv_pair = fd_x[:,0]
print(type(first_row_tv_pair[0]))