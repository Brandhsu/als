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


def pca_importance(dataset):
    components = dataset.pca_.components_
    n_components = components.shape[0]
    feature_names = dataset.feature_names_
    
    largest_component = [np.abs(components[i]).argmax() for i in range(n_components)]
    largest_components_names = [feature_names[largest_component[i]] for i in range(n_components)]
    
    df = {'pc-{}'.format(i+1): largest_components_names[i] for i in range(n_components)}
    
    return pd.DataFrame(df.items(), columns=['pc', 'feature'])