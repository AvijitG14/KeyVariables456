# KeyVariables456

This is a machine learning project intended to test the different methods to find the most effective method to determine the key variables in a dataset (that is, which method will lead to the greatest accuracy over multiple runs). It will use Kevin Arvai and Carlos Borroto's Genetic Variant Classification dataset available on Kaggle, in which each row represents a chromosome that is likely to have conflicting classifications based on its own attributes. This dataset has over 65,000 rows and 46 columns. The y-value in this set is a Class column that detects the presence of conflicting clinical classification, and it is a discrete value that only accepts binary values 0 and 1.

One aspect of note with this dataset is that there are null values for a few attributes among all its rows. We will have a two-step process for handling the null values: we will first ignore them to see if they affect the weights of the others, and will later incorporate them into the final results in the case that other weights are affected in their absence. This dataset contains both discrete and continuous attributes: Chromosome, Alternate Chromosome, and Origin represent the values that can be one from a finite set, while Position, Allele Frequency, and Disease Name represent the values that can be one from an infinitely large set.

# Collaborators:

Juan Espinoza

Avijit Gupta

Mark Tan

# Algorithms Used:

Logistic Regression

Support Vector Machine

Convolution Neural Networking
