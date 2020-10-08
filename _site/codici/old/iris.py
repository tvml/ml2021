#!/usr/bin/env python
# -*- coding: utf-8 -*-

# visualizzazione features di dataset iris
# sulla diagonale, distribuzione dei valori della feature nel training set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as st
import seaborn as sns

# definisce un vettore di colori
colors = sns.color_palette("husl", 4)
# dichiara alcune proprietà grafiche della figura
sns.set(style="darkgrid", context='paper', palette=colors, rc={"figure.figsize": (16, 8),'image.cmap': 'jet', 'lines.linewidth':.7})

def scatter_matrix(df):
    features=[ x for x in df.columns if x!='class']
    classes=[ x for x in df['class'].unique()]
    nclasses=len(classes)
    nfeatures=len(features)
    data=np.array([np.array([np.array(df[df['class']==c][f]) for f in features]) for c in classes])
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    for i in range(nfeatures):
        flattened= np.array([val for sublist in data[:,i] for val in sublist])
        x = np.arange(min(flattened), max(flattened), .001)
        for j in range(nfeatures):
            ax = fig.add_subplot(nfeatures,nfeatures, i+nfeatures*j+1, axisbg="#F8F8F8")
            plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
            if i==j:
                y = st.gaussian_kde(flattened)(x)
                ax.plot(x, y, color=colors[3], alpha=.7)
                ax.fill_between(x, 0, y, color=colors[3], alpha=0.5)
                plt.xlim(min(flattened), max(flattened))
            else:
                for c in range(nclasses):
                    ax.scatter(data[c][i],data[c][j], color=colors[c],alpha=.9, s=10)
                plt.xlim(min(flattened), max(flattened))
            if i==0:
                ax.set_ylabel(features[j], fontsize=10)
            if j==nfeatures-1:
                ax.set_xlabel(features[i], fontsize=10)
    handles = [mpatches.Patch(color=colors[k], label=classes[k]) for k in range(nclasses)]
    plt.figlegend(handles, classes, 'upper center', ncol=nclasses, labelspacing=0. , fontsize=12)
    plt.show()

data_folder = '.'

df = pd.read_csv(data_folder+'iris.csv', delimiter=';')
scatter_matrix(df)