# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
from scipy import ndimage
import scipy.io as io
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'


# definisce un vettore di colori
colors = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c', '#137e6d', '#be0119', '#3b638c', '#af6f09']

pal = mpl.colors.ListedColormap(['#8b8b8b', '#be0119'], name='from_list', N=None)

data = io.loadmat('20news_w100')
documents = data['documents']
documents = documents.toarray().T
newsgroups = data['newsgroups'][0]

n_docs, n_words = documents.shape
n_groups =np.unique(newsgroups).shape[0]
#sort documents by number of words and choose the first n_sel
n_sel = 2000
chosen_docs_arg = np.argsort(np.sum(documents, axis=1))
chosen_docs_arg = chosen_docs_arg[::-1][:n_sel]
chosen_docs = documents[chosen_docs_arg]
chosen_docs_newsgroups = newsgroups[chosen_docs_arg]

#sort by newsgroups label
sorted_arg = np.argsort(chosen_docs_newsgroups)
chosen_docs = chosen_docs[sorted_arg]
chosen_docs_newsgroups = chosen_docs_newsgroups[sorted_arg]

#zoom the image to show it
image = ndimage.zoom(chosen_docs, (1, 1))

fig=plt.figure(figsize=(16,8))
ax = fig.gca()
ax.imshow(image, cmap=pal, interpolation='none', alpha=.5)
#draw a red line betweent different newsgroups
groups_label = np.unique(chosen_docs_newsgroups)
for i in range(len(groups_label) - 1):
    y = np.where(chosen_docs_newsgroups == groups_label[i + 1])[0]
    y = y[0]
    ax.plot([y]*n_words, color=colors[8], lw=2)
plt.axis('tight')
plt.xlabel('words')
plt.ylabel('documents')
plt.xticks(range(0, n_words+1, 10), range(0, n_words+1, 10))
plt.yticks(range(0, n_sel+1, 100), range(0, n_sel+1, 100))
plt.show()