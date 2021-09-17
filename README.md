# README #

### What is this repository for? ###
* This is a pure python implementation that makes predictions of ratings using collaborative filtering.
* This implementation experiments with intagrating predictive analytical models.
* This is for demonstation purposes only.  Pure python is no good for big data :)
* Version 0.0.1

### What does this code do ###
* It is designed to be a framework to perform collaborative filtering
* It predicts ratings for users based on other similar users.
* Finds simlar target get data
* Measures their similarty using Pearson's Correlation, Euclidean Distance, Cosine Simularity, or Jaccard Simularty.
* Returns an overall RMSD (RMSE) score for the algorithm.
* Predicts a rating based on a particular user using several different algorithms
* Predicts a rating user by integrating a neural network (yes, there are more suitable algorithms) 

### How to use ###
* See the main.py for a demonstration of its use (it takes time!)

### Included data ###
* Snippets of data was taken from the Yelp challenge (user information is kept private)

### TODO LIST ###
* Add a gradient boosting algorithm (neural nets are not ideal)
* Integrate with my own neural network implementation