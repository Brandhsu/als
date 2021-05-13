import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


def plot_feature_importances(df):
    '''
    Adapted from https://github.com/WillKoehrsen/feature-selector
    '''
    #Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    #Normalise the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    #Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10,6))
    ax = plt.subplot()
    #Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
           df['importance_normalized'].head(15),
           align = 'center', edgecolor = 'k')
    #Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    #Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importance')
    plt.show()
    return df


def pca_importance(dataset, pca_):
    components = pca_.components_
    n_components = components.shape[0]
    feature_names = dataset.feature_names_
    
    largest_component = [np.abs(components[i]).argmax() for i in range(n_components)]
    largest_components_names = [feature_names[largest_component[i]] for i in range(n_components)]
    
    df = {'pc-{}'.format(i+1): largest_components_names[i] for i in range(n_components)}
    
    return pd.DataFrame(df.items(), columns=['pc', 'feature']), largest_component # returning dataframe and component indices


# https://www.kaggle.com/aviv10/als-meaningful-clusters-and-feature-extraction#Functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

def important_clusters(XClusterLabels, Y, numClusters, threshold = 0.8, labelOfInterest = 1):
    ''' Check which clusters express a given class label in a ratio greater than a threshold
    
    Arguments:
        XClusterLabels: ndarray of shape (n_samples,) cluster predictions for the training data
        Y: ndarray of shape (n_samples,), training labels
        numCluster: an integer representing the number of clusters
        threshold: a float representing the ratio threshold for a cluster to be significant, defaults to 0.8
        labelOfInterest: an integer representing the class label of interest
    
    Returns:
        An ndarray containing 0 (not exceeding the threshold) or 1 (exceeding the threshold) for each cluster,
        and an ndarray containing the ratio for each cluster
    '''
    meaningfulList = np.zeros((numClusters))
    ratioList = np.zeros((numClusters))
    
    for i in np.arange(numClusters):
        YClusterLabels = Y[XClusterLabels == i]
        ratio = YClusterLabels[YClusterLabels == labelOfInterest].shape[0] / YClusterLabels.shape[0]
        if ratio >= threshold:
            meaningfulList[i] = 1
        ratioList[i] = ratio
    return meaningfulList, ratioList


def extract_important_features(dataset, X, XClusterLabels, clusterOfInterest, numFeatures=5000, visualize=False):
    ''' Find which features are important in a random forest classifier with two classes: 
    being in the cluster of interest, and not being in it.

    Arguments:
        X: Pandas DataFrame containing the training data
        XClusterLabels: ndarray of shape (n_samples,) cluster predictions for the training data
        clusterOfInterest: an integer representing the cluster of interest
        numFeatures: an integer representing the number of important features to return, defaults to 5000
        visualize: a boolean representing whether to visualize the important features, defaults to False

    Returns:
        A Pandas DataFrame containing the top numFeatures most important features
    '''
    
    clf = RandomForestClassifier()
    newClusterLabels = np.zeros(XClusterLabels.shape)
    newClusterLabels[XClusterLabels == clusterOfInterest] = 1
    clf.fit(X, newClusterLabels)

    feature_importance_values = clf.feature_importances_
    features = list(dataset.feature_names_)
    feature_importances = pd.DataFrame({'feature': features, 'importance':feature_importance_values})
    return plot_feature_importances(feature_importances)[:numFeatures]


def feature_importance(dataset):
    # Replace X_train with embedding later
    X_train = dataset.xtr
    y_train = dataset.ytr
    
    numClusters=10
    kmeans = KMeans(n_clusters=numClusters).fit(X_train)
    XclusterLabels = kmeans.predict(X_train)
    meaningList, ratioList = important_clusters(XclusterLabels, y_train, numClusters, threshold = 0.9)

    for i in np.arange(numClusters):
        if meaningList[i] == 1:
            test = extract_important_features(dataset, X_train, XclusterLabels, i, visualize=True)
            print(test[:10])