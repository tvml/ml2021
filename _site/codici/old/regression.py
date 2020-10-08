# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Regressione

# +
from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline
# -

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV, KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, RidgeCV
import seaborn.apionly as sns
import copy

# +
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

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
cmap_big = cm.get_cmap('Spectral', 512)
cmap = mcolors.ListedColormap(cmap_big(np.linspace(0.7, 0.95, 256)))

bbox_props = dict(boxstyle="round,pad=0.3", fc=colors[0], alpha=.5)
# -

# # Esame del dataset Housing

# Features:
#     
# <pre>
# 1. CRIM      per capita crime rate by town
# 2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.
# 3. INDUS     proportion of non-retail business acres per town
# 4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 5. NOX       nitric oxides concentration (parts per 10 million)
# 6. RM        average number of rooms per dwelling
# 7. AGE       proportion of owner-occupied units built prior to 1940
# 8. DIS       weighted distances to five Boston employment centres
# 9. RAD       index of accessibility to radial highways
# 10. TAX      full-value property-tax rate per $10,000
# 11. PTRATIO  pupil-teacher ratio by town
# 12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 13. LSTAT    % lower status of the population
# 14. MEDV     Median value of owner-occupied homes in $1000s
# </pre>

# Lettura del dataset in dataframe pandas

df = pd.read_csv('../dataset/housing.data.txt', header=None, sep='\s+')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.shape

# ## Visualizzazione delle caratteristiche del dataset

# Matrice delle distribuzioni mutue delle feature. Sulla diagonale, distribuzione delle singole feature

# +
cols = ['LSTAT', 'RM', 'INDUS', 'AGE', 'MEDV']

fig = plt.figure(figsize=(16, 8))
sns.pairplot(df[cols], height=4, diag_kind='kde', 
             plot_kws=dict(color=colors[8]), 
             diag_kws=dict(shade=True, alpha=.7, color=colors[0]))
plt.show()
# -

# Visualizzazione della matrice di correlazione. Alla posizione $(i,j)$ il coefficiente di correlazione (lineare) tra le feature $i$ e $j$. Valore in $[-1,1]$: $1$ correlazione perfetta, $-1$ correlazione inversa perfetta, $0$ assenza di correlazione

cm = np.corrcoef(df[cols].values.T)
plt.figure(figsize=(12,4))
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 10},
                 yticklabels=cols,
                 xticklabels=cols,
                 cmap = cmap)
plt.tight_layout()
plt.show()



# ### Regressione di MEDV rispetto a una sola feature

print("Feature utilizzabili: {0}".format(', '.join(map(str, df.columns[:-1]))))

mi = mutual_info_regression(df[df.columns[:-1]], df[df.columns[-1]])
dmi = pd.DataFrame(mi, index=df.columns[:-1], columns=['mi']).sort_values(by='mi', ascending=False)
dmi.head(20)

# Utilizza la feature più significativa

feat = dmi.index[0]

X = df[[feat]].values
y = df['MEDV'].values

y

results = []

# Regressione lineare standard: la funzione di costo è $$C(\mathbf{w})=\frac{1}{2}\sum_i (y(\mathbf{w},\mathbf{x}_i) - t_i)^2$$

# crea modello di regressione lineare
r = LinearRegression()
# ne apprende i coefficienti sui dati disponibili
r = r.fit(X, y)

# Misura di qualità utilizzata: 
# - MSE (Errore quadratico medio) definito come $$\frac{1}{n}\sum_{i=1}^n (y(\mathbf{w},\mathbf{x}_i) - t_i)^2$$

p = r.predict(X)
# valuta MSE su dati e previsioni
mse = mean_squared_error(p,y)

mse

print('w0: {0:.3f}, w1: {1:.3f}, MSE: {2:.3f}'.format(r.intercept_, r.coef_[0],mse))

x = np.linspace(min(X),max(X),100).reshape(-1,1)
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor="xkcd:light grey")
plt.plot(x, r.predict(x), color=colors[2]) 
plt.xlabel(feat)
plt.ylabel('MEDV')
plt.title('Regressione su una feature', fontsize=16)
plt.text(0.85, 0.9, 'MSE: {0:.3f}'.format(mse), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.show()

# Valuta il modello su test set al fine di evitare overfitting

# partiziona dataset in training (80%) e test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Crea una pipeline con il solo modello di regressione

pipe = Pipeline([('regression', LinearRegression())])
pipe = pipe.fit(X_train, y_train)
p_train = pipe.predict(X_train)
p_test = pipe.predict(X_test)
mse_train = mean_squared_error(p_train,y_train)
mse_test = mean_squared_error(p_test,y_test)

r = pipe.named_steps['regression']
print('w0: {0:.3f}, w1: {1:.3f}, MSE-train: {2:.3f}, MSE-test: {3:.3f}'.format(r.intercept_, r.coef_[0],mse_train, mse_test))

results.append(['Regression, 1 feature', mse_train, mse_test])

x = np.linspace(min(X),max(X),100).reshape(-1,1)
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X_train, y_train, c=colors[8], edgecolor="xkcd:light grey", label='Train')
plt.scatter(X_test, y_test, c=colors[0], edgecolor='black', label='Test')
plt.plot(x, pipe.predict(x), color=colors[2]) 
plt.xlabel(feat)
plt.ylabel('MEDV')
plt.title('Regressione su una feature con test set', fontsize=16)
plt.text(0.9, 0.9, 'MSE\ntrain {0:.3f}\ntest {1:.3f}'.format(mse_train, mse_test), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.show()

# Aggiungi standardizzazione della feature, modificandone i valori in modo da ottenere media $0$ e varianza $1$. Utilizza le pipeline di scikit-learn per definire una sequenza di task: in questo caso i dati sono normalizzati mediante uno StandardScaler e sui risultati viene applicato il modello di regressione.

# +
pipe = Pipeline([('scaler', StandardScaler()),('regression', LinearRegression())])
pipe = pipe.fit(X_train, y_train)

p_train = pipe.predict(X_train)
p_test = pipe.predict(X_test)
mse_train = mean_squared_error(p_train,y_train)
mse_test = mean_squared_error(p_test,y_test)
# -

s = pipe.named_steps['scaler']
print('Scaling: mean: {0:.3f}, var: {1:.3f}, scale: {2:.3f}'.format(s.mean_[0], s.var_[0],s.scale_[0]))

r = pipe.named_steps['regression']
print('w0: {0:.3f}, w1: {1:.3f}, MSE-train: {2:.3f}, MSE-test: {3:.3f}'.format(r.intercept_, r.coef_[0],mse_train, mse_test))

results.append(['Regression, 1 feature, scaled', mse_train, mse_test])

x = np.linspace(min(X),max(X),100).reshape(-1,1)
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X_train, y_train, c=colors[8], edgecolor='xkcd:light grey', label='Train')
plt.scatter(X_test, y_test, c=colors[0], edgecolor='black', label='Test')
plt.plot(x, pipe.predict(x), color=colors[2]) 
plt.xlabel(feat)
plt.ylabel('MEDV')
plt.text(0.9, 0.9, 'MSE\ntrain {0:.3f}\ntest {1:.3f}'.format(mse_train, mse_test), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.title('Regressione su una feature standardizzata, con test set', fontsize=16)
plt.show()

# La valutazione potrebbe dipendere eccessivamente dalla coppia training-test set (varianza). 
# Utilizzo della cross validation per valutare il modello. Si applica un KFold per suddividere il training set $X$ in n_splits coppie (training set, test set)

pipe = Pipeline([('scaler', StandardScaler()),('regression', LinearRegression())])
k_fold = KFold(n_splits=3)
mse = []
preds = []
# itera su tutte le coppie (training set - test set)
for train, test in k_fold.split(X):
    # effettua l'apprendimento dei coefficienti sul training set
    r = pipe.fit(X[train], y[train])
    # appende in una lista il modello di regressione appreso
    preds.append(copy.deepcopy(r))
    mse.append(mean_squared_error(r.predict(X[test]),y[test]))
for i,r in enumerate(preds):
    c = [r.named_steps['scaler'].scale_[0], r.named_steps['scaler'].mean_[0], r.named_steps['regression'].intercept_, r
                  .named_steps['regression'].coef_[0]]
    print('Fold: {0:2d}, mean:{1:.3f}, scale: {2:.3f}, w0: {3:.3f}, w1: {4:.3f}, MSE test set: {5:.3f}'.format(i, c[0],c[1],c[2],c[3],mse[i]))
# restituisce le medie dei coefficienti e del MSE su tutti i fold
print('\nMSE - media: {0:.3f}, dev.standard: {1:.3f}'.format(np.mean(mse), np.std(mse)))

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey')
for i, r in enumerate(preds):
    plt.plot(X, r.predict(X), color=colors[i%7], linewidth=1) 
plt.xlabel(feat)
plt.ylabel('MEDV')
plt.title('Regressione su una feature standardizzata, con CV', fontsize=16)
plt.show()

# Utilizza la funzione cross_val_score di scikit-learn per effettuare la cross validation

p = Pipeline([('scaler', StandardScaler()),('regression', LinearRegression())])
# apprende il modello su tutto il training set
r = p.fit(X, y)
# calcola costo derivante dall'applicazione del modello su tutto il dataset, quindi con possibile overfitting
mse = mean_squared_error(r.predict(X),y)
# effettua la cross validation, derivando il costo sul test set per tutti i fold
scores = cross_val_score(estimator=p, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
# calcola costo medio su tutti i fold
mse_cv = -scores.mean()

results.append(['Regression, 1 feature, scaled, CV', mse, mse_cv])

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey')
plt.plot(X, r.predict(X), color=colors[2]) 
plt.xlabel(feat)
plt.ylabel('MEDV')
plt.title('Regressione su una feature standardizzata, con CV', fontsize=16)
plt.text(0.88, 0.9, 'MSE\ntrain {0:.3f}\nmedia CV {1:.3f}'.format(mse, mse_cv), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.show()

# ### Regressione con regolazione

# Utilizza un modello con regolazione L1 (Lasso): la funzione di costo è $$C(\mathbf{w})=\frac{1}{2}\sum_i ((y(\mathbf{w},\mathbf{x}_i) - t_i)^2+\frac{\alpha}{2}\sum_j |w_j|$$ 

#fissa un valore per l'iperparametro
alpha = 0.5
p = Pipeline([('scaler', StandardScaler()),('regression', Lasso(alpha=alpha))])
r = p.fit(X, y)
mse = mean_squared_error(r.predict(X),y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
mse_cv = -scores.mean()

results.append(['Regression L1, 1 feature, scaled, CV, alpha=0.5', mse, mse_cv])

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey')
plt.plot(X, r.predict(X), color=colors[2]) 
plt.xlabel(feat)
plt.ylabel('MEDV')
plt.title(r'Regressione lineare con regolazione L1 ($\alpha={0:.2f}$)'.format(alpha), fontsize=16)
plt.text(0.88, 0.9, 'MSE\ntrain {0:.3f}\nmedia CV {1:.3f}'.format(mse, mse_cv), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.show()

# Applica un modello con regolazione L2 (Ridge): la funzione di costo è $$C(\mathbf{w})=\frac{1}{2}\sum_i ((y(\mathbf{w},\mathbf{x}_i) - t_i)^2+\frac{\alpha}{2}\sum_j w_j^2$$

#fissa un valore per l'iperparametro
alpha = 0.5
p = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha=alpha))])
r = p.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
mse = mean_squared_error(r.predict(X),y)
mse_cv = -scores.mean()

results.append(['Regression L2, 1 feature, scaled, CV, alpha=0.5', mse, mse_cv])

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey')
plt.plot(X, r.predict(X), color=colors[2]) 
plt.xlabel('Numero medio di locali [RM]')
plt.ylabel('Prezzo in migliaia di $ [MEDV]')
plt.title(r'Regressione lineare con regolazione L2 ($\alpha={0:.2f}$)'.format(alpha), fontsize=16)
plt.text(0.88, 0.9, 'MSE\ntrain {0:.3f}\nmedia CV {1:.3f}'.format(mse, mse_cv), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.show()

# Applica un modello con regolazione Elastic Net: la funzione di costo è $$C(\mathbf{w})=\frac{1}{2}\sum_i ((y(\mathbf{w},\mathbf{x}_i) - t_i)^2+\frac{\alpha}{2}(\gamma\sum_j |w_j|+(1-\gamma)\sum_j w_j^2)$$

alpha = 0.5
gamma = 0.3
p = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha=alpha, l1_ratio=gamma))])
r = p.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
mse = mean_squared_error(r.predict(X),y)
mse_cv = -scores.mean()

results.append(['Regression Elastic Net, 1 feature, scaled, CV, alpha=0.5, gamma=0.3', mse, mse_cv])

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey')
plt.plot(X, r.predict(X), color=colors[2]) 
plt.xlabel('Numero medio di locali [RM]')
plt.ylabel('Prezzo in migliaia di $ [MEDV]')
plt.title(r'Regressione lineare con regolazione Elastic Net ($\alpha={0:.2f}, \gamma={1:.2f}$)'.format(alpha, gamma), fontsize=16)
plt.text(0.88, 0.9, 'MSE\ntrain {0:.3f}\nmedia CV {1:.3f}'.format(mse, mse_cv), fontsize=12, transform=ax.transAxes, 
         bbox=bbox_props)
plt.show()

# ## Funzioni base polinomiali

# Regressione lineare standard con funzioni base polinomiali. Utilizza PolynomialFeatures di scikit-learn, che implementa funzioni base polinomiali fino al grado dato

deg = 3
pipe_regr = Pipeline([('scaler', StandardScaler()),('bf', PolynomialFeatures(degree=deg)),('regression', LinearRegression())])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
mse = mean_squared_error(r.predict(X),y)
mse_cv = -scores.mean()

results.append(['Regression, Polynomial, 1 feature, scaled, degree={0:d}, CV'.format(deg), mse, mse_cv])

xmin = np.floor(min(X)[0])
xmax = np.ceil(max(X)[0])
x = np.linspace(xmin,xmax,100).reshape(-1, 1)
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey')
plt.plot(x, r.predict(x), color=colors[2]) 
plt.xlabel('Numero medio di locali [RM]')
plt.ylabel('Prezzo in migliaia di $ [MEDV]')
plt.title(r'Regressione lineare con f.b. polinomiali ($d={0:3d}$)'.format(deg), fontsize=16)
plt.text(0.88, 0.9, 'MSE\ntrain {0:.3f}\nmedia CV {1:.3f}'.format(mse, mse_cv), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.show()

# Visualizzazione dei residui: differenze $y_i-t_i$ in funzione di $y_i$

# +
y_pred = r.predict(X)

mm = min(y_pred)
mx = max(y_pred)

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(y_pred,  (y_pred - y), c=colors[8], edgecolor='xkcd:light grey')
plt.xlabel(r'Valori predetti ($y_i$)')
plt.ylabel(r'Residui ($y_i-t_i$)')
plt.hlines(y=0, xmin=(int(mm)/10)*10, xmax=(int(mx)/10)*10+10, color=colors[2], lw=2)
plt.text(0.88, 0.9, 'MSE: d = {0:d}\ntrain {1:.3f}\nmedia CV {2:.3f}'.format(deg, mse, mse_cv), fontsize=12, transform=ax.transAxes, 
         bbox=bbox_props)
plt.show()
# -

res = []
for deg in range(1,30):
    r = Pipeline([('scaler', StandardScaler()),('bf', PolynomialFeatures(degree=deg)),('regression', LinearRegression())]).fit(X, y)
    scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
    mse = mean_squared_error(r.predict(X),y)
    mse_cv = -scores.mean()
    res.append([deg, mse, mse_cv])

top = 15
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot([r[0] for r in res[:top]],  [r[1] for r in res[:top]], color=colors[8],label=r'Train') 
plt.plot([r[0] for r in res[:top]],  [r[2] for r in res[:top]], color=colors[2],label=r'Test') 
l=plt.legend()

alpha = 1
deg = 3
pipe_regr = Pipeline([('scaler', StandardScaler()),('bf', PolynomialFeatures(degree=deg)),('regression', Lasso(alpha=alpha))])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')

mse = mean_squared_error(r.predict(X),y)
mse_cv = -scores.mean()
xmin = np.floor(min(X)[0])
xmax = np.ceil(max(X)[0])
x = np.linspace(xmin,xmax,100).reshape(-1, 1)
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor='white')
plt.plot(x, r.predict(x), color=colors[2]) 
plt.xlabel('Numero medio di locali [RM]')
plt.ylabel('Prezzo in migliaia di $ [MEDV]')
plt.title(r'Regressione lineare con f.b. polinomiali e regolazione L2 ($d={0:3d}, \alpha={1:.3f}$)'.format(deg, alpha), fontsize=16)
plt.text(0.88, 0.9, 'MSE\ntrain {0:.3f}\nmedia CV {1:.3f}'.format(mse, mse_cv), fontsize=12, transform=ax.transAxes, 
         bbox=bbox_props)
plt.show()

res = []
for deg in range(1,20):
    r = Pipeline([('scaler', StandardScaler()),('bf', PolynomialFeatures(degree=deg)),('regression', Lasso(alpha=alpha))]).fit(X, y)
    scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
    mse = mean_squared_error(r.predict(X),y)
    mse_cv = -scores.mean()
    res.append([deg, mse, mse_cv])

top = 15
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot([r[0] for r in res[:top]],  [r[1] for r in res[:top]], color=colors[8],label=r'Train') 
plt.plot([r[0] for r in res[:top]],  [r[2] for r in res[:top]], color=colors[2],label=r'Test') 
l=plt.legend()

# +
y_pred = r.predict(X)

mm = min(y_pred)
mx = max(y_pred)

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(y_pred,  (y_pred - y),c=colors[8], edgecolor='white',label='Train')
plt.xlabel(r'Valori predetti ($y_i$)')
plt.ylabel(r'Residui ($y_i-t_i$)')
plt.hlines(y=0, xmin=(int(mm)/10)*10, xmax=(int(mx)/10)*10+10, color=colors[2], lw=2)
plt.tight_layout()
plt.show()
# -

# ## Regressione su tutte le feature

X = df[df.columns[:-1]]
y = df[df.columns[-1]]

r = LinearRegression()
r.fit(X, y)
print('MSE: {0:.3f}'.format(mean_squared_error(r.predict(X),y)))

# +
y_pred = r.predict(X)

mm = min(y_pred)
mx = max(y_pred)

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(y_pred,  (y_pred - y), c=colors[8], edgecolor='xkcd:light grey')
plt.xlabel(r'Valori predetti ($y_i$)')
plt.ylabel(r'Residui ($y_i-t_i$)')
plt.hlines(y=0, xmin=(int(mm)/10)*10, xmax=(int(mx)/10)*10+10, color=colors[2], lw=2)
plt.show()
# -

r = Pipeline([('scaler', StandardScaler()),('regression', LinearRegression())])
r.fit(X, y)
print('MSE: {0:.3f}'.format(mean_squared_error(r.predict(X),y)))

# +
y_pred = r.predict(X)

mm = min(y_pred)
mx = max(y_pred)

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(y_pred,  (y_pred - y), c=colors[8], edgecolor='xkcd:light grey')
plt.xlabel(r'Valori predetti ($y_i$)')
plt.ylabel(r'Residui ($y_i-t_i$)')
plt.hlines(y=0, xmin=(int(mm)/10)*10, xmax=(int(mx)/10)*10+10, color=colors[2], lw=2)
plt.show()
# -

# Applica cross-validation

r = Pipeline([('scaler', StandardScaler()),('regression', LinearRegression())])
scores = cross_val_score(estimator=r, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
print('MSE')
print(-scores)
print('media {0:.3f}, dev.standard {1:.3f}'.format(-scores.mean(), -scores.std()))

alpha = 0.5
r = Pipeline([('scaler', StandardScaler()),('regression', Lasso(alpha=alpha))])
scores = cross_val_score(estimator=r, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
print('MSE')
print(-scores)
print('media {0:.3f}, dev.standard {1:.3f}'.format(-scores.mean(), -scores.std()))

alpha = 10
r = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha=alpha))])
scores = cross_val_score(estimator=r, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
print('MSE')
print(-scores)
print('media {0:.3f}, dev.standard {1:.3f}'.format(-scores.mean(), -scores.std()))

alpha = 0.5
gamma = 0.3
r = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha=alpha, l1_ratio=gamma))])
scores = cross_val_score(estimator=r, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
print('MSE')
print(-scores)
print('media {0:.3f}, dev.standard {1:.3f}'.format(-scores.mean(), -scores.std()))

# LassoCV effettua la ricerca del miglior valore per $\alpha$

pipe_regr = Pipeline([('scaler', StandardScaler()),('regression', LassoCV(cv=7))])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
best_alpha = pipe_regr.named_steps['regression'].alpha_
print(r'Miglior valore di alpha: {0:.3f}'.format(best_alpha))
print('MSE: {0:.3f}'.format(-scores.mean()))

pipe_regr = Pipeline([('scaler', StandardScaler()),('regression', Lasso(alpha = best_alpha))])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('MSE: {0:.3f}'.format(-scores.mean()))

# +
y_pred = r.predict(X)

mm = min(y_pred)
mx = max(y_pred)

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(y_pred,  (y_pred - y), c=colors[8], edgecolor='xkcd:light grey')
plt.xlabel(r'Valori predetti ($y_i$)')
plt.ylabel(r'Residui ($y_i-t_i$)')
plt.hlines(y=0, xmin=(int(mm)/10)*10, xmax=(int(mx)/10)*10+10, color=colors[2], lw=2)
plt.tight_layout()
plt.show()
# -

pipe_regr = Pipeline([('scaler', StandardScaler()),('regression', RidgeCV(cv=20))])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
best_alpha = pipe_regr.named_steps['regression'].alpha_
print(r'Miglior valore di alpha: {0:.3f}'.format(best_alpha))
print('MSE: {0:.3f}'.format(-scores.mean()))

pipe_regr = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha = best_alpha))])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('MSE: {0:.3f}'.format(-scores.mean()))

# +
r = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha=best_alpha))]).fit(X, y)

y_pred = r.predict(X)

mm = min(y_pred)
mx = max(y_pred)

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(y_pred,  (y_pred - y), c=colors[8], edgecolor='xkcd:light grey')
plt.xlabel(r'Valori predetti ($y_i$)')
plt.ylabel(r'Residui ($y_i-t_i$)')
plt.hlines(y=0, xmin=(int(mm)/10)*10, xmax=(int(mx)/10)*10+10, color=colors[2], lw=2)
plt.tight_layout()
plt.show()
# -

# ## Model selection

X = np.array(df[df.columns[:-1]])
y = np.array(df[df.columns[-1]])

# ### Lasso

# Ricerca su griglia di valori per alpha in Lasso

domain = np.linspace(0,10,100)
cv = 10
scores = []
kf = KFold(n_splits=cv)
# considera tutti i valori di alpha in domain
for a in domain:
    # definisce modello con Lasso
    p = Pipeline([('scaler', StandardScaler()),('regression', Lasso(alpha=a))])
    xval_err = 0
    # per ogni coppia train-test valuta l'errore sul test set del modello istanziato sulla base del training set
    for k, (train_index, test_index) in enumerate(kf.split(X,y)):
        p.fit(X[train_index], y[train_index])
        y1 = p.predict(X[test_index])
        err = y1 - y[test_index]
        xval_err += np.dot(err,err)
    # calcola erroe medio 
    score = xval_err/X.shape[0]
    scores.append([a,score])
scores = np.array(scores)

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot(scores[:,0], scores[:,1]) 
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.title(r'MSE al variare di $\alpha$ in Lasso')
plt.show()

min_index = np.argmin(scores[:,1])
print('Miglior valore per alpha: {0:.5f}. MSE={1:.3f}'.format(scores[min_index,0], scores[min_index,1]))

# Utilizzo di GridSearchCV

# +
domain = np.linspace(0,10,100)
param_grid = [{'regression__alpha': domain}]
p = Pipeline([('scaler', StandardScaler()),('regression', Lasso())])

clf = GridSearchCV(p, param_grid, cv=5, scoring='neg_mean_squared_error')
clf = clf.fit(X,y)
sc = -clf.cv_results_['mean_test_score']
# -

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot(domain,sc) 
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.title(r'MSE al variare di $\alpha$ in Lasso')
plt.show()

min_index = np.argmin(sc)
print('Miglior valore per alpha: {0:.5f}. MSE={1:.3f}'.format(domain[min_index], sc[min_index]))

# Utilizzo di LassoCV, che ricerca il miglior valore di $\alpha$ valutando lo score su un insieme di possibili valori mediante cross validation. 

domain=np.linspace(0,10,100)
p = Pipeline([('scaler', StandardScaler()),('regression', LassoCV(cv=10, alphas=domain))])
r = p.fit(X, y)
scores = np.mean(r.named_steps['regression'].mse_path_, axis=1)

plt.figure(figsize=(16, 8))
plt.plot(r.named_steps['regression'].alphas_, scores)
plt.xlabel(r'$\alpha$')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()

best_alpha = r.named_steps['regression'].alpha_
print(r'Miglior valore di alpha: {0:.5f}'.format(best_alpha))
i, = np.where(r.named_steps['regression'].alphas_ == best_alpha)
print('MSE: {0:.5f}'.format(scores[i][0]))

r.named_steps['regression'].coef_

# Valuta Lasso con il valore trovato per $\alpha$ sull'intero dataset

p = Pipeline([('scaler', StandardScaler()),('regression', Lasso(alpha = best_alpha))])
scores = cross_val_score(estimator=p, X=X, y=y, cv=20, scoring='neg_mean_squared_error')
print('MSE: {0:.3f}'.format(-scores.mean()))

# ### Ridge

# Ricerca su griglia di valori per alpha in Ridge

domain = np.linspace(80,120,100)
cv = 10
scores = []
kf = KFold(n_splits=cv)
for a in domain:
    p = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha=a))])
    xval_err = 0
    for k, (train_index, test_index) in enumerate(kf.split(X,y)):
        p.fit(X[train_index], y[train_index])
        y1 = p.predict(X[test_index])
        err = y1 - y[test_index]
        xval_err += np.dot(err,err)
    score = xval_err/X.shape[0]
    scores.append([a,score])
scores = np.array(scores)

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot(scores[:,0], scores[:,1]) 
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.title(r'MSE al variare di $\alpha$ in Ridge')
plt.show()

min_index = np.argmin(scores[:,1])
best_alpha = scores[min_index,0]
print('Miglior valore per alpha: {0:.5f}. MSE={1:.3f}'.format(scores[min_index,0], scores[min_index,1]))

# Applica sul dataset con il valore trovato per $\alpha$

p = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha = best_alpha))])
r = p.fit(X, y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('alpha: {0:.3f}, MSE: {1:.3f}'.format(best_alpha, -scores.mean()))

# Utilizzo di GridSearchCV

# +
domain = np.linspace(80,120,100)
param_grid = [{'regression__alpha': domain}]
p = Pipeline([('scaler', StandardScaler()),('regression', Ridge())])

clf = GridSearchCV(p, param_grid, cv=10, scoring='neg_mean_squared_error')
clf = clf.fit(X,y)
scores = -clf.cv_results_['mean_test_score']
# -

fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot(domain,scores) 
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.title(r'MSE al variare di $\alpha$ in Ridge')
plt.show()

min_index = np.argmin(scores)
print('Miglior valore per alpha: {0:.5f}. MSE={1:.3f}'.format(domain[min_index], scores[min_index]))

# Applica sul dataset con il valore trovato per $\alpha$

p = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha = best_alpha))])
r = p.fit(X, y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('alpha: {0:.3f}, MSE: {1:.3f}'.format(best_alpha, -scores.mean()))

# Utilizza RidgeCV, che ricerca il miglior valore di $\alpha$ valutando lo score su un insieme di possibili valori mediante cross validation

domain = np.linspace(0.1, 10, 100)
p = Pipeline([('scaler', StandardScaler()),('regression', RidgeCV(alphas=domain, store_cv_values = True))])
r = p.fit(X, y)
scores = np.mean(r.named_steps['regression'].cv_values_, axis=0)

plt.figure(figsize=(16, 8))
plt.plot(domain, scores)
plt.xlabel(r'$\alpha$')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()

best_alpha = p.named_steps['regression'].alpha_
print(r'Miglior valore di alpha: {0:.6f}'.format(best_alpha))
i, = np.where(domain == best_alpha)
print('score: {0:.3f}'.format(scores[i][0]))

# Valuta Ridge con il valore trovato per  α
#   sull'intero dataset

p = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha = best_alpha))])
r = p.fit(X, y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('alpha: {0:.3f}, MSE: {1:.3f}'.format(best_alpha, -scores.mean()))

r.named_steps['regression'].coef_

# ### Elastic net

# Ricerca su griglia 2d di valori per $\alpha$ e $\gamma$

scores = []
for a in np.linspace(0,1,10):
    for l in np.linspace(0,1,10):
        p = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha=a, l1_ratio=l))])
        score = cross_val_score(estimator=p, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
        scores.append([a,l,-score.mean()])

scores = np.array(scores)
min_index = np.argmin(scores[:,2])
best_alpha = scores[min_index, 0]
best_gamma = scores[min_index, 1]
print(r"Migliore coppia: alpha={0:.2f}, gamma={1:.2f}. MSE={2:.3f}".format(best_alpha,best_gamma, scores[min_index,2]))


p = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha = best_alpha, l1_ratio=best_gamma))])
r = p.fit(X, y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('alpha: {0:.3f}, gamma: {1:.3f}; MSE: {2:.3f}'.format(best_alpha, best_gamma, -scores.mean()))

# Utilizza GridsearchCV

# +
param_grid = [{'regression__alpha': np.linspace(0,1,10), 'regression__l1_ratio': np.linspace(0,1,10)}]
p = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha=alpha, l1_ratio=gamma))])

clf = GridSearchCV(p, param_grid, cv=5, scoring='neg_mean_squared_error')
clf = clf.fit(X,y)
sc = -clf.cv_results_['mean_test_score']
# -

best_alpha = clf.best_params_['regression__alpha']
best_gamma = clf.best_params_['regression__l1_ratio']
print(r"Migliore coppia: alpha={0:.2f}, gamma={1:.2f}. MSE={2:.3f}".format(best_alpha,
                                        best_gamma, -clf.best_score_))

p = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha = best_alpha, l1_ratio=best_gamma))])
r = p.fit(X, y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('alpha: {0:.3f}, gamma: {1:.3f}; MSE: {2:.3f}'.format(best_alpha, best_gamma, -scores.mean()))


