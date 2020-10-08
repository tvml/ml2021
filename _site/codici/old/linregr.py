# -*- coding: utf-8 -*-
# Visualizzazione dell'andamento della funzione di errore quadratico nella regressione
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# +
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
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd:goldenrod', 'xkcd:cadet blue', 
          'xkcd:scarlet']
# -

# definisce un vettore di colori
colors = sns.color_palette("husl", 4)
# dichiara alcune proprietà grafiche della figura
sns.set(style="darkgrid", context='paper', palette=colors, rc={"figure.figsize": (16, 8),'image.cmap': 'jet', 'lines.linewidth':.7})

# legge i dati in dataframe pandas
data = pd.read_csv("../dataset/cars.csv", delimiter=',', header=0, names=['X','y'])

# calcola dimensione dei dati
n = len(data)

# visualizza dati mediante scatter
fig = plt.figure()
fig.patch.set_facecolor('white')
ax = fig.gca()
ax.scatter(data['X'], data['y'], s=40,c='r', marker='o', alpha=.5)
plt.xlabel(u'Velocità in mph', fontsize=14)
plt.ylabel('Distanza di arresto in ft', fontsize=14)
plt.show()

# Estrae dal dataframe l'array X delle features e aggiunge ad esso una colonna di 1
X=np.array(data['X']).reshape(-1,1)
X = np.column_stack((np.ones(n), X))

# Estrae dal dataframe l'array t dei valori target
t=np.array(data['y']).reshape(-1,1)

# mostra distribuzione dell'errore quadratico medio al variare dei coefficienti

# insieme dei valori considerati per i coefficienti
w0_list = np.linspace(-100, 100, 100)
w1_list = np.linspace(-100, 100, 100)

# crea una griglia di coppie di valori 
w0, w1 = np.meshgrid(w0_list, w1_list)

# definisce la funzione da calcolare in ogni punto della griglia
def error(v1, v2):
    theta = np.array((v1, v2)).reshape(-1, 1)
    e=(np.dot(X,theta)-t)
    return np.dot(e.T,e)[0,0]/(2*n)

v_error=np.vectorize(error)
e=v_error(w0,w1).T


fig = plt.figure()
fig.patch.set_facecolor('white')
ax = fig.gca(projection='3d')
surf=ax.plot_surface(w0, w1, e, rstride=1, cstride=1, cmap=plt.cm.jet , linewidth=0, antialiased=True)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8)
plt.xlabel(r"$w_0$", fontsize=12)
plt.ylabel(r"$w_1$", fontsize=12)
plt.title(r"Errore quadratico medio al variare dei coefficienti $w_0,w_1$", fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.jet)
plt.show()

fig = plt.figure(figsize=(12,12))
fig.patch.set_facecolor('white')
ax = fig.gca()
im = plt.imshow(e, origin='lower', extent=(w0_list.min(),w0_list.max(),w1_list.min(), w1_list.max()), aspect='auto',alpha=.8)
#plt.contour(w0, w1, e,color='r', lw=0.7)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
plt.xlabel(r"$w_0$", fontsize=12)
plt.ylabel(r"$w_1$", fontsize=12)
plt.title(r"Errore quadratico medio al variare dei coefficienti $w_0,w_1$", fontsize=12)
fig.colorbar(im, shrink=0.5, aspect=7, cmap=plt.cm.jet)
plt.show()


