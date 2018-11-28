import pandas as pd

'''
    This program performs basic access of elements within
    the dataset of the CSV file.
'''
#process csv file that we will use for project
dataframe = pd.read_csv('~/Downloads/clinvar_conflicting.csv')
print(dataframe.dtypes)

#convert CSV dataframe into numpy array for easy data manipulation
info = [dataframe.iloc[i,:] for i in range(dataframe.shape[0])]
final_data = np.array(info)

fd_x = np.delete(final_data, 18, 1)
fd_y = final_data[:,18:19]

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