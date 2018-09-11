# PREPROCESSING DATA
# The act of formatting data in a certain way for machine learning algorithms

import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3], [-1.2, 7.8, -6.1], [3.9, 0.4, 2.1], [7.3, -9.9, -4.5]])

# Binarization preprocessing technique
# Transforms all numerical values into booleans, with threshold indicating what value is 1 (above) or 0 (below)
def Binarize():
    data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
    print("\nBinarized data:\n", data_binarized)

# Mean removal preprocessing technique
# Removes the mean so each feature is centered on zero
def MeanRemove():
    print("\nBEFORE:")
    print("Mean:", input_data.mean(axis=0))
    print("Std deviation:", input_data.std(axis=0))

    data_scaled = preprocessing.scale(input_data)

    print("\nAFTER")
    print("Mean:", data_scaled.mean(axis=0))
    print("Std deviation:", data_scaled.std(axis=0))

# Scaling preprocessing technique
# Transforms numerical values by taking using the smallest and biggest to then convert all into a relative (feature) range 
def Scale():
    data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data_scaled = data_scaler.fit_transform(input_data)
    print("\nScaled data:\n", data_scaled)

# Normalization preprocessing technique
# L1 normalization: least absolute deviations, makes Σ of absolute values in each row is 1
# L2 normalization: least squares deviations, makes Σ of squares values in wach row is 1
def Normalize():
    data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
    data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
    print("\nL1 normalized data:\n", data_normalized_l1)
    print("\nL2 normalized data:\n", data_normalized_l2)

# User select of preprocessing technique
ppt = input("Preprocessing techniques:\n1) Binarization\n2) Mean removal\n3) Scaling\n4) Normalization\nSelect: ")
if ppt == "1":
    Binarize()
elif ppt == "2":
    MeanRemove()
elif ppt == "3":
    Scale()
elif ppt == "4":
    Normalize()