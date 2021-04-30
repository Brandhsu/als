# [End ALS Kaggle Challenge](https://www.kaggle.com/alsgroup/end-als)

## Task
*Determine and discover features (transcriptomes or gene mutations) that could cause ALS.*

## Dataset
* Processed transcriptomics data from DESeq2
    * Data is nonlinear 
    * Many features, small amount of data
    * Class imbalance 

## Model
*The main goal is to train a semi-supervised model on ALS patients to determine features that best correspond with the disease.*

This is done in a couple of steps:
1. Train a multi-loss autoencoder with reconstruction and supervised loss
2. Extract the latent features of the network and run a clustering algorithm on it
3. Determine the semantics of each cluster (e.g. comparing it with class labels)
4. Learn the most important features for each cluster

Visualization of steps
![alt text](https://github.com/Brandhsu/als/assets/architecture.jpg)

## Notes
* als_notebook.ipynb includes an autoencoder neural network and some ML/Stats stuff at the bottom
* models trained with small amt. of data and many parameters struggle to learn even on train-set
    * this is suspected to be due to the "curse of dimensionality" and nonconvexity
    * model has too many solutions and thus struggles to find the optimal parameters

## Implemented
* Data loading (DESeq2)
* NN and ML model cross-validation/training, evaluating, visualizing
* Dimensionality reduction 
    * PCA
    * Feature Selection


