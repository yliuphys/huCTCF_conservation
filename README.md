# huCTCF_conservation
A deep learning model for predicting the cell type conservation of human CTCF binding

The CTCF dataset we used included data from 40 cell types (PMID: 26257180).  The degree of cell type conservation of CTCF binding at a given genomic locus was expressed as mcv, which indicated the number of cell types in which the binding was detected.  We selected CTCF binding regions with the top 10% (most conserved) and bottom 10% (least conserved) mcv, which had mcv >= 36 and <=4, respectively, and used these data to build a 1D Convolutional Neural Network (CNN) to predict the degree of cell type conservation based on DNA sequences at the loci. We used Keras of Python library to apply sequential neural network. The Keras application program interface was running on top of TensorFlow.

Prerequisite: Python 3; numpy; pandas; scikit-learn; tensorflow; keras.

Input format: the deep learning model requires the following two file as input:
Data: data.txt file is a tab separate, 2 column, plain text file. The first column is 151bp DNA sequence of human CTCF region harbor major allele of SNP, and second column is the same sequence but harbor minor allele of SNP.
Pre-build 1DCNN model: the h5 file (model_151bp_1DCNN_after_tunning_032119.h5) for the model can be downloaded from Github python_code folder.

Output format: a tab separated plain text file with 4 columns. The first two columns are identical from the input data (data.txt) and the last two columns are the prediction results shown with probability for major and minor allele CTCF to be labelled as the most conserved CTCF binding region.

Main function to get the final results:
python CNN_CTCF.py data.txt 

Test the model with the test dataset:
Make sure the CNN_CTCF.py, model_151bp_1DCNN_after_tunning_032119.h5, test_data.txt are in the same folder.
Run prediction with: python CNN_CTCF.py test_data.txt

