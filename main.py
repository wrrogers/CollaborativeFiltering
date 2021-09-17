from Collaborative_Filtering import Collaborative_Filtering as cf
import pandas as pd
import os

df = pd.read_hdf('University_Heights_reviews.h5')

#df = pd.read_csv(pwd + 'i2iCF2.csv')  

r = cf(df)
k = 3

#print("\n")
print("THE RMSD USING THE PEARSONS CORRELATION: ", r.rmsd(algo = "Pearsons", topk = k))

#print("\n")
print("THE RMSD FOR PEARSON LINLI: ", r.bf_rmsd(algo = "Pearsons", topk = k))
#print("THE RMSD FOR EUCLIDEAN:     ", r.bf_rmsd(algo = "Euclidean"    , topk = k))
#print("THE RMSD FOR COSINE:        ", r.bf_rmsd(algo = "Cosine"       , topk = k))
#print("THE RMSD FOR JACCARD:        ", r.bf_rmsd(algo = "Jaccard"       , topk = k))

#Use this to build a trainingset to predict ratings using machine learning
trainingset = r.build_train(filename = 'train.csv', algo = 'Pearson_Will')
print("THE RMSD FOR THE NN:        ", r.fit_nn(epochs = 1600))
