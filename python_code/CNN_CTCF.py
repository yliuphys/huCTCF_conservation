#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:36:05 2019

@author: yong
"""

#the input file has only two column, with first column as 151bp human CTCF region harbor major allele, and second column as 151bp CTCF region harbor minor allele 

import sys
import numpy as np
import pandas as pd


filename = sys.argv[1]

#read the CTCF file
ctcf_SNP = pd.read_table(filename)

#organization the data into a format that the deep learning algorithm can use
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#the LabelEncoder encodes a sequence of bases as a sequence of integers
integer_encoder = LabelEncoder()
#the OneHotEncoder converts an array of intergers to a sparse matrix where
#each row corresponding to one possible value of each feature
one_hot_encoder = OneHotEncoder(n_values = 4)

ctcf_majorAllele_SNP_input_features = []

#creat a list of sequences
sequences = list(ctcf_SNP.iloc[:,0])

for sequence in sequences:
  integer_encoded = integer_encoder.fit_transform(list(sequence))
  integer_encoded = np.array(integer_encoded).reshape(-1, 1)
  one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
  ctcf_majorAllele_SNP_input_features.append(one_hot_encoded.toarray())
  
  
#similarly, we can encode the minor allele
ctcf_minorAllele_SNP_input_features = []

#creat a list of sequences
sequences = list(ctcf_SNP.iloc[:,1])
sequences = [x.upper() for x in sequences] #convert to uppercase
sequences = [x[0:151] for x in sequences] #get 151bp region



for sequence in sequences:
  integer_encoded = integer_encoder.fit_transform(list(sequence))
  integer_encoded = np.array(integer_encoded).reshape(-1, 1)
  one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
  ctcf_minorAllele_SNP_input_features.append(one_hot_encoded.toarray())  
  
from tensorflow import keras  


model = keras.models.load_model("model_151bp_1DCNN_after_tunning_032119.h5")  

major_allele_predicted_labels = model.predict(np.stack(ctcf_majorAllele_SNP_input_features))
major_allele_predicted_labels_df = pd.DataFrame({'major_allele_predict_label1':major_allele_predicted_labels[:,1]})

minor_allele_predicted_labels = model.predict(np.stack(ctcf_minorAllele_SNP_input_features))
minor_allele_predicted_labels_df = pd.DataFrame({'minor_allele_predict_label1':minor_allele_predicted_labels[:,1]})


ctcf_SNP_withPrediction = pd.concat([ctcf_SNP,major_allele_predicted_labels_df], axis =1)
ctcf_SNP_withPrediction = pd.concat([ctcf_SNP_withPrediction,minor_allele_predicted_labels_df], axis =1)

ctcf_SNP_withPrediction.to_csv("results.csv")


 
