# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="WzGezE-jLT-Q"
# # Regressione

# + colab={} colab_type="code" id="HEOGBYJ_LT-X"
from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline

# + colab={} colab_type="code" id="OYZrd4k1LT-n"
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
import seaborn as sns
import copy

# + colab={} colab_type="code" id="K_KPvIWPLgGH"
import urllib.request

filepath = "../dataset/"
url = "https://tvml.github.io/ml1920/dataset/"

def get_file(filename,local):
    if local:
        return filepath+filename
    else:
        urllib.request.urlretrieve (url+filename, filename)
        return filename


# + colab={} colab_type="code" id="e81w6WCDLT-1"
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

plt.style.use('ggplot')

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

# + [markdown] colab_type="text" id="A1W_8vkLLT_A"
# # Esame del dataset Housing

# + [markdown] colab_type="text" id="Sx_cuu0CLT_D"
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

# + [markdown] colab_type="text" id="yi836kY3LT_F"
# Lettura del dataset in dataframe pandas

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="zsM2ILnPLT_K" outputId="e862d1f4-3ec6-4ba1-cd6a-348300107059"
df = pd.read_csv(get_file('housing.data.txt',local=1), header=None, sep='\s+')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.shape

# + [markdown] colab_type="text" id="IHWGWS1WLT_S"
# ## Visualizzazione delle caratteristiche del dataset

# + [markdown] colab_type="text" id="S_hYwNBHLT_U"
# Matrice delle distribuzioni mutue delle feature. Sulla diagonale, distribuzione delle singole feature

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="BXuF5r3aLT_X" outputId="13034d4b-2ed1-445b-bc9b-50dfc19c4fa7"
cols = ['LSTAT', 'RM', 'INDUS', 'AGE', 'MEDV']

fig = plt.figure(figsize=(16, 8))
sns.pairplot(df[cols], height=4, diag_kind='kde', 
             plot_kws=dict(color=colors[8]), 
             diag_kws=dict(shade=True, alpha=.7, color=colors[0]))
plt.show()

# + [markdown] colab_type="text" id="ubZe7F0GLT_e"
# Visualizzazione della matrice di correlazione. Alla posizione $(i,j)$ il coefficiente di correlazione (lineare) tra le feature $i$ e $j$. Valore in $[-1,1]$: $1$ correlazione perfetta, $-1$ correlazione inversa perfetta, $0$ assenza di correlazione

# + colab={"base_uri": "https://localhost:8080/", "height": 513} colab_type="code" id="7oBXiAU_LT_g" outputId="f609eba8-8bd8-4716-c516-049e4bf8745c"
cm = np.corrcoef(df[cols].values.T)
plt.figure(figsize=(14,7))
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

# + [markdown] colab_type="text" id="XrFgg_2ALT_s"
# ### Regressione di MEDV rispetto a una sola feature

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="tqqNLcLeLT_t" outputId="a36f40c0-bed0-40e5-8151-0c3caf49e271"
print("Feature utilizzabili: {0}".format(', '.join(map(str, df.columns[:-1]))))

# + colab={"base_uri": "https://localhost:8080/", "height": 452} colab_type="code" id="FinaEtbQLT_y" outputId="71eba5c5-5d5d-45b5-fd41-7a869124c834"
mi = mutual_info_regression(df[df.columns[:-1]], df[df.columns[-1]])
dmi = pd.DataFrame(mi, index=df.columns[:-1], columns=['mi']).sort_values(by='mi', ascending=False)
dmi.head(20)

# + [markdown] colab_type="text" id="xH3Vn74sLT_3"
# Utilizza la feature più significativa

# + colab={} colab_type="code" id="Y5j9EGzjLT_7"
feat = dmi.index[0]

# + colab={"base_uri": "https://localhost:8080/", "height": 419} colab_type="code" id="-YxDJENWMRZQ" outputId="6560daea-48f3-4938-a187-1bbca93ada5e"
df[[feat,'MEDV']]

# + colab={} colab_type="code" id="_T4GidP4LT_-"
X = df[[feat]].values
y = df['MEDV'].values

# + colab={"base_uri": "https://localhost:8080/", "height": 799} colab_type="code" id="F5a2R3YPLUAC" outputId="c5851d4b-e841-4283-884e-5ce1d49df6ad"
y

# + colab={} colab_type="code" id="en1TYfxoLUAI"
results = []

# + [markdown] colab_type="text" id="D1vfvAeELUAN"
# Regressione lineare standard: la funzione di costo è $$C(\mathbf{w})=\frac{1}{2}\sum_i (y(\mathbf{w},\mathbf{x}_i) - t_i)^2$$

# + colab={} colab_type="code" id="tnEU119hLUAO"
# crea modello di regressione lineare
r = LinearRegression()
# ne apprende i coefficienti sui dati disponibili
r = r.fit(X, y)

# + [markdown] colab_type="text" id="usLU4_fJLUAU"
# Misure di qualità utilizzate: 
# - MSE (Errore quadratico medio) definito come $$\frac{1}{n}\sum_{i=1}^n (y(\mathbf{w},\mathbf{x}_i) - t_i)^2$$
#
# - $r^2$ (Coefficiente di determinazione) definito come frazione di varianza dei valori target spiegata dalla regressione $$\frac{\sum_{i=1}^n (y(\mathbf{w},\mathbf{x}_i) - \overline{t})^2}{\sum_{i=1}^n (t_i - \overline{t})^2}=1-\frac{\sum_{i=1}^n (y(\mathbf{w},\mathbf{x}_i) - t_i)^2}{\sum_{i=1}^n (t_i - \overline{t})^2}$$
#
# dove $$\overline{t}=\frac{1}{n}\sum_{i=1}^nt_i$$ è il valor medio del target

# + colab={} colab_type="code" id="WTlCxUcELUAV"
p = r.predict(X)
# valuta MSE su dati e previsioni
mse = mean_squared_error(p,y)
r2 = r2_score(p,y)

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="6sBzDNYyLUAa" outputId="7f0aeb77-32b3-41d4-96af-808448d65603"
print('w0: {0:.3f}, w1: {1:.3f}, MSE: {2:.3f}, r2={3:5.2f}'.format(r.intercept_, r.coef_[0],mse, r2))

# + colab={"base_uri": "https://localhost:8080/", "height": 542} colab_type="code" id="EkPAp_wXLUAn" outputId="9acfd5f3-70d0-4658-a501-92174dea0890"
x = np.linspace(min(X),max(X),100).reshape(-1,1)
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor="xkcd:light grey")
plt.plot(x, r.predict(x), color=colors[2]) 
plt.xlabel(feat)
plt.ylabel('MEDV')
plt.title('Regressione su una feature', fontsize=16)
plt.text(0.85, 0.9, 'MSE: {0:.3f}'.format(mse), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.text(0.85, 0.85, 'r2: {0:.3f}'.format(r2), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.show()

# + [markdown] colab_type="text" id="RvAjhuG4LUBI"
# Valuta il modello su test set al fine di evitare overfitting

# + colab={} colab_type="code" id="FuBJSthcLUBJ"
# partiziona dataset in training (80%) e test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# + [markdown] colab_type="text" id="KC4SwMQALUBM"
# Crea una pipeline con il solo modello di regressione

# + colab={} colab_type="code" id="yUKCri3vLUBN"
pipe = Pipeline([('regression', LinearRegression())])
pipe = pipe.fit(X_train, y_train)
p_train = pipe.predict(X_train)
p_test = pipe.predict(X_test)
mse_train = mean_squared_error(p_train,y_train)
mse_test = mean_squared_error(p_test,y_test)

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="rcuh3OXlLUBQ" outputId="ec499eb5-0de6-4fba-f60d-0d1395efc00d"
r = pipe.named_steps['regression']
print('w0: {0:.3f}, w1: {1:.3f}, MSE-train: {2:.3f}, MSE-test: {3:.3f}'.format(r.intercept_, r.coef_[0],mse_train, mse_test))

# + colab={} colab_type="code" id="KYnfa4nILUBY"
results.append(['Regression, 1 feature', mse_train, mse_test])

# + colab={"base_uri": "https://localhost:8080/", "height": 542} colab_type="code" id="8udR0PE0LUBb" outputId="e97141e9-bfff-4cbe-a5cc-796409c1d3da"
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

# + [markdown] colab_type="text" id="SWiPKre4LUBf"
# Aggiungi standardizzazione della feature, modificandone i valori in modo da ottenere media $0$ e varianza $1$. Utilizza le pipeline di scikit-learn per definire una sequenza di task: in questo caso i dati sono normalizzati mediante uno StandardScaler e sui risultati viene applicato il modello di regressione.

# + colab={} colab_type="code" id="Gf_exd6iLUBg"
pipe = Pipeline([('scaler', StandardScaler()),('regression', LinearRegression())])
pipe = pipe.fit(X_train, y_train)

p_train = pipe.predict(X_train)
p_test = pipe.predict(X_test)
mse_train = mean_squared_error(p_train,y_train)
mse_test = mean_squared_error(p_test,y_test)

# + colab={} colab_type="code" id="I-k_EpmNLUBi" outputId="d7329b99-cde7-4eb0-c80e-2ddbdb99e746"
s = pipe.named_steps['scaler']
print('Scaling: mean: {0:.3f}, var: {1:.3f}, scale: {2:.3f}'.format(s.mean_[0], s.var_[0],s.scale_[0]))

# + colab={} colab_type="code" id="Bb2GUAiSLUBk" outputId="fa179b8d-d169-4425-af93-1e8d907e41b9"
r = pipe.named_steps['regression']
print('w0: {0:.3f}, w1: {1:.3f}, MSE-train: {2:.3f}, MSE-test: {3:.3f}'.format(r.intercept_, r.coef_[0],mse_train, mse_test))

# + colab={} colab_type="code" id="2UfPjFAhLUBo"
results.append(['Regression, 1 feature, scaled', mse_train, mse_test])

# + colab={} colab_type="code" id="GKrltf3iLUBq" outputId="0621d9de-b9c2-4c45-bb4d-cf133687eceb"
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

# + [markdown] colab_type="text" id="v4lO8GN0LUBs"
# La valutazione potrebbe dipendere eccessivamente dalla coppia training-test set (varianza). 
# Utilizzo della cross validation per valutare il modello. Si applica un KFold per suddividere il training set $X$ in n_splits coppie (training set, test set)

# + colab={} colab_type="code" id="5AfkUitELUBt" outputId="94668fd6-bc73-469a-ddfa-ab3cbce544c7"
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

# + colab={} colab_type="code" id="W1wE1Iu_LUBx" outputId="e3c04be7-194f-4684-945d-17f823631597"
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey')
for i, r in enumerate(preds):
    plt.plot(X, r.predict(X), color=colors[i%7], linewidth=1) 
plt.xlabel(feat)
plt.ylabel('MEDV')
plt.title('Regressione su una feature standardizzata, con CV', fontsize=16)
plt.show()

# + [markdown] colab_type="text" id="IqSLytR5LUBz"
# Utilizza la funzione cross_val_score di scikit-learn per effettuare la cross validation

# + colab={} colab_type="code" id="iOROy93FLUB0"
p = Pipeline([('scaler', StandardScaler()),('regression', LinearRegression())])
# apprende il modello su tutto il training set
r = p.fit(X, y)
# calcola costo derivante dall'applicazione del modello su tutto il dataset, quindi con possibile overfitting
mse = mean_squared_error(r.predict(X),y)
# effettua la cross validation, derivando il costo sul test set per tutti i fold
scores = cross_val_score(estimator=p, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
# calcola costo medio su tutti i fold
mse_cv = -scores.mean()

# + colab={} colab_type="code" id="JeOw11joLUB1"
results.append(['Regression, 1 feature, scaled, CV', mse, mse_cv])

# + colab={} colab_type="code" id="slet7vExLUB4" outputId="822559c9-7510-456b-f1a2-a1dd2f892d59"
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey')
plt.plot(X, r.predict(X), color=colors[2]) 
plt.xlabel(feat)
plt.ylabel('MEDV')
plt.title('Regressione su una feature standardizzata, con CV', fontsize=16)
plt.text(0.88, 0.9, 'MSE\ntrain {0:.3f}\nmedia CV {1:.3f}'.format(mse, mse_cv), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.show()

# + [markdown] colab_type="text" id="F-DAAHg2LUB6"
# ### Regressione con regolazione

# + [markdown] colab_type="text" id="bQ_q5KSlLUB7"
# Utilizza un modello con regolazione L1 (Lasso): la funzione di costo è $$C(\mathbf{w})=\frac{1}{2}\sum_i ((y(\mathbf{w},\mathbf{x}_i) - t_i)^2+\frac{\alpha}{2}\sum_j |w_j|$$ 

# + colab={} colab_type="code" id="fRDmrJQxLUB7"
#fissa un valore per l'iperparametro
alpha = 0.5
p = Pipeline([('scaler', StandardScaler()),('regression', Lasso(alpha=alpha))])
r = p.fit(X, y)
mse = mean_squared_error(r.predict(X),y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
mse_cv = -scores.mean()

# + colab={} colab_type="code" id="mUA7XrpuLUB_"
results.append(['Regression L1, 1 feature, scaled, CV, alpha=0.5', mse, mse_cv])

# + colab={} colab_type="code" id="v6Oc9VaHLUCB" outputId="c10e5797-28b8-457e-995c-46b8cd249600"
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey')
plt.plot(X, r.predict(X), color=colors[2]) 
plt.xlabel(feat)
plt.ylabel('MEDV')
plt.title(r'Regressione lineare con regolazione L1 ($\alpha={0:.2f}$)'.format(alpha), fontsize=16)
plt.text(0.88, 0.9, 'MSE\ntrain {0:.3f}\nmedia CV {1:.3f}'.format(mse, mse_cv), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.show()

# + [markdown] colab_type="text" id="shnWSlL9LUCE"
# Applica un modello con regolazione L2 (Ridge): la funzione di costo è $$C(\mathbf{w})=\frac{1}{2}\sum_i ((y(\mathbf{w},\mathbf{x}_i) - t_i)^2+\frac{\alpha}{2}\sum_j w_j^2$$

# + colab={} colab_type="code" id="Ie5Kz85vLUCE"
#fissa un valore per l'iperparametro
alpha = 0.5
p = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha=alpha))])
r = p.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
mse = mean_squared_error(r.predict(X),y)
mse_cv = -scores.mean()

# + colab={} colab_type="code" id="Wesq6ZKwLUCG"
results.append(['Regression L2, 1 feature, scaled, CV, alpha=0.5', mse, mse_cv])

# + colab={} colab_type="code" id="uw9K3q-cLUCJ" outputId="a3b0e46f-e910-416c-db66-ad4974c356ac"
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey')
plt.plot(X, r.predict(X), color=colors[2]) 
plt.xlabel('Numero medio di locali [RM]')
plt.ylabel('Prezzo in migliaia di $ [MEDV]')
plt.title(r'Regressione lineare con regolazione L2 ($\alpha={0:.2f}$)'.format(alpha), fontsize=16)
plt.text(0.88, 0.9, 'MSE\ntrain {0:.3f}\nmedia CV {1:.3f}'.format(mse, mse_cv), fontsize=12, transform=ax.transAxes, bbox=bbox_props)
plt.show()

# + [markdown] colab_type="text" id="OqfmXidYLUCL"
# Applica un modello con regolazione Elastic Net: la funzione di costo è $$C(\mathbf{w})=\frac{1}{2}\sum_i ((y(\mathbf{w},\mathbf{x}_i) - t_i)^2+\frac{\alpha}{2}(\gamma\sum_j |w_j|+(1-\gamma)\sum_j w_j^2)$$

# + colab={} colab_type="code" id="GRV4hATlLUCL"
alpha = 0.5
gamma = 0.3
p = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha=alpha, l1_ratio=gamma))])
r = p.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
mse = mean_squared_error(r.predict(X),y)
mse_cv = -scores.mean()

# + colab={} colab_type="code" id="u3nqNp2rLUCR"
results.append(['Regression Elastic Net, 1 feature, scaled, CV, alpha=0.5, gamma=0.3', mse, mse_cv])

# + colab={} colab_type="code" id="nCil7cUBLUCT" outputId="d4078f6f-46c9-47a2-a3fc-6619ba83a3bf"
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

# + [markdown] colab_type="text" id="ENzFUFjtLUCX"
# ## Funzioni base polinomiali

# + [markdown] colab_type="text" id="ta6XLw-TLUCX"
# Regressione lineare standard con funzioni base polinomiali. Utilizza PolynomialFeatures di scikit-learn, che implementa funzioni base polinomiali fino al grado dato

# + colab={} colab_type="code" id="NYZHfioDLUCY"
deg = 3
pipe_regr = Pipeline([('scaler', StandardScaler()),('bf', PolynomialFeatures(degree=deg)),('regression', LinearRegression())])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
mse = mean_squared_error(r.predict(X),y)
mse_cv = -scores.mean()

# + colab={} colab_type="code" id="ugTU8bQzLUCZ"
results.append(['Regression, Polynomial, 1 feature, scaled, degree={0:d}, CV'.format(deg), mse, mse_cv])

# + colab={} colab_type="code" id="8F9ibDnALUCc" outputId="8cf7a996-4acb-4d7c-8052-2e0d3f4fff90"
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

# + [markdown] colab_type="text" id="N79DgIiOLUCd"
# Visualizzazione dei residui: differenze $y_i-t_i$ in funzione di $y_i$

# + colab={} colab_type="code" id="ZOINwoDRLUCe" outputId="f46b202c-efa0-4a7b-aa13-ac35485f91c9"
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

# + colab={} colab_type="code" id="4wp9Da1fLUCj"
res = []
for deg in range(1,30):
    r = Pipeline([('scaler', StandardScaler()),('bf', PolynomialFeatures(degree=deg)),('regression', LinearRegression())]).fit(X, y)
    scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
    mse = mean_squared_error(r.predict(X),y)
    mse_cv = -scores.mean()
    res.append([deg, mse, mse_cv])

# + colab={} colab_type="code" id="QCGdDVDMLUCl" outputId="d7935e4e-1f92-4932-c9fc-08e615bff938"
top = 15
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot([r[0] for r in res[:top]],  [r[1] for r in res[:top]], color=colors[8],label=r'Train') 
plt.plot([r[0] for r in res[:top]],  [r[2] for r in res[:top]], color=colors[2],label=r'Test') 
l=plt.legend()

# + colab={} colab_type="code" id="y783lyGCLUCn"
alpha = 1
deg = 3
pipe_regr = Pipeline([('scaler', StandardScaler()),('bf', PolynomialFeatures(degree=deg)),('regression', Lasso(alpha=alpha))])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')

# + colab={} colab_type="code" id="5CSq6xHBLUCp" outputId="a5e385da-8460-43f8-a564-907d5f821865"
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

# + colab={} colab_type="code" id="_cVYm4ScLUCr"
res = []
for deg in range(1,20):
    r = Pipeline([('scaler', StandardScaler()),('bf', PolynomialFeatures(degree=deg)),('regression', Lasso(alpha=alpha))]).fit(X, y)
    scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
    mse = mean_squared_error(r.predict(X),y)
    mse_cv = -scores.mean()
    res.append([deg, mse, mse_cv])

# + colab={} colab_type="code" id="nx-9VU6GLUCs" outputId="c56a17f4-7f3c-4fea-f509-abe2abd36518"
top = 15
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot([r[0] for r in res[:top]],  [r[1] for r in res[:top]], color=colors[8],label=r'Train') 
plt.plot([r[0] for r in res[:top]],  [r[2] for r in res[:top]], color=colors[2],label=r'Test') 
l=plt.legend()

# + colab={} colab_type="code" id="YUsFDVWnLUCu" outputId="45cb8555-3eca-4e6a-a3ca-332b8806dbca"
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

# + [markdown] colab_type="text" id="1rQvtlkDLUCw"
# ## Regressione su tutte le feature

# + colab={} colab_type="code" id="wCa0n7SALUCw"
X = df[df.columns[:-1]]
y = df[df.columns[-1]]

# + colab={} colab_type="code" id="-ZpdPPIhLUC2" outputId="7355cac9-6187-4c66-d037-923e78d972dd"
r = LinearRegression()
r.fit(X, y)
print('MSE: {0:.3f}'.format(mean_squared_error(r.predict(X),y)))

# + colab={} colab_type="code" id="7NT6s1lFLUC3" outputId="68b8cd2f-ad26-4d5c-ddb4-5069d3e1bfdd"
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

# + colab={} colab_type="code" id="_OmD8ysmLUC5" outputId="bbbdab3a-b8c8-4777-9879-5ec547e30e15"
r = Pipeline([('scaler', StandardScaler()),('regression', LinearRegression())])
r.fit(X, y)
print('MSE: {0:.3f}'.format(mean_squared_error(r.predict(X),y)))

# + colab={} colab_type="code" id="0H4WoOOaLUDE" outputId="00495aac-f972-4a2f-cdd0-fc41eb8d57fd"
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

# + [markdown] colab_type="text" id="Hx9up98BLUDK"
# Applica cross-validation

# + colab={} colab_type="code" id="_-IUj46qLUDK" outputId="b8da2178-3cf8-4524-bf3e-d08f75b53a7e"
r = Pipeline([('scaler', StandardScaler()),('regression', LinearRegression())])
scores = cross_val_score(estimator=r, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
print('MSE')
print(-scores)
print('media {0:.3f}, dev.standard {1:.3f}'.format(-scores.mean(), -scores.std()))

# + colab={} colab_type="code" id="N_Qz2U0kLUDL" outputId="0664b93e-f1c5-40aa-db76-bcc3f8d72a17"
alpha = 0.5
r = Pipeline([('scaler', StandardScaler()),('regression', Lasso(alpha=alpha))])
scores = cross_val_score(estimator=r, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
print('MSE')
print(-scores)
print('media {0:.3f}, dev.standard {1:.3f}'.format(-scores.mean(), -scores.std()))

# + colab={} colab_type="code" id="kXD6FQ9WLUDO" outputId="10295334-286d-407c-efc8-03009a770891"
alpha = 10
r = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha=alpha))])
scores = cross_val_score(estimator=r, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
print('MSE')
print(-scores)
print('media {0:.3f}, dev.standard {1:.3f}'.format(-scores.mean(), -scores.std()))

# + colab={} colab_type="code" id="6aEYp8AqLUDR" outputId="e8380447-b4de-4635-d998-8117de2951e5"
alpha = 0.5
gamma = 0.3
r = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha=alpha, l1_ratio=gamma))])
scores = cross_val_score(estimator=r, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
print('MSE')
print(-scores)
print('media {0:.3f}, dev.standard {1:.3f}'.format(-scores.mean(), -scores.std()))

# + [markdown] colab_type="text" id="wWyNt6n7LUDV"
# LassoCV effettua la ricerca del miglior valore per $\alpha$

# + colab={} colab_type="code" id="lJFg0U9dLUDW" outputId="e71324af-d537-4b72-d981-d8a405b3b688"
pipe_regr = Pipeline([('scaler', StandardScaler()),('regression', LassoCV(cv=7))])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
best_alpha = pipe_regr.named_steps['regression'].alpha_
print(r'Miglior valore di alpha: {0:.3f}'.format(best_alpha))
print('MSE: {0:.3f}'.format(-scores.mean()))

# + colab={} colab_type="code" id="fiPDuI5jLUDY" outputId="f808b282-bc17-4b03-edf1-bf719d1d5a62"
pipe_regr = Pipeline([('scaler', StandardScaler()),('regression', Lasso(alpha = best_alpha))])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('MSE: {0:.3f}'.format(-scores.mean()))

# + colab={} colab_type="code" id="GnHnxy_tLUDg" outputId="1ce740e8-957a-4f99-8cf4-190d0499d979"
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

# + colab={} colab_type="code" id="lgR0sJKnLUDj" outputId="026486fc-82d0-4e83-af3b-cda09902d3b5"
pipe_regr = Pipeline([('scaler', StandardScaler()),('regression', RidgeCV(cv=20))])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
best_alpha = pipe_regr.named_steps['regression'].alpha_
print(r'Miglior valore di alpha: {0:.3f}'.format(best_alpha))
print('MSE: {0:.3f}'.format(-scores.mean()))

# + colab={} colab_type="code" id="S1NpK9TLLUDl" outputId="2a7cad10-0553-496a-fe62-149205e674d1"
pipe_regr = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha = best_alpha))])
r = pipe_regr.fit(X, y)
scores = cross_val_score(estimator=r, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('MSE: {0:.3f}'.format(-scores.mean()))

# + colab={} colab_type="code" id="5NIn5i7FLUDo" outputId="c8ddd6fb-97aa-44fa-9a03-c5c4002c6938"
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

# + [markdown] colab_type="text" id="nj5oUjitLUDr"
# ## Model selection

# + colab={} colab_type="code" id="OJjzVL2qLUDr"
X = np.array(df[df.columns[:-1]])
y = np.array(df[df.columns[-1]])

# + [markdown] colab_type="text" id="9NYkpIB1LUDt"
# ### Lasso

# + [markdown] colab_type="text" id="qtl5jriNLUDt"
# Ricerca su griglia di valori per alpha in Lasso

# + colab={} colab_type="code" id="96YUVTefLUDu"
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

# + colab={} colab_type="code" id="kAlV30iWLUDv" outputId="41734d87-1e42-48f7-cfcd-35b778522c3b"
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot(scores[:,0], scores[:,1]) 
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.title(r'MSE al variare di $\alpha$ in Lasso')
plt.show()

# + colab={} colab_type="code" id="2d8x05mbLUEA" outputId="0a41d0de-456d-4ea3-80ab-6e2e39860265"
min_index = np.argmin(scores[:,1])
print('Miglior valore per alpha: {0:.5f}. MSE={1:.3f}'.format(scores[min_index,0], scores[min_index,1]))

# + [markdown] colab_type="text" id="XjkUwEG8LUEC"
# Utilizzo di GridSearchCV

# + colab={} colab_type="code" id="xQ90VoqGLUED"
domain = np.linspace(0,10,100)
param_grid = [{'regression__alpha': domain}]
p = Pipeline([('scaler', StandardScaler()),('regression', Lasso())])

clf = GridSearchCV(p, param_grid, cv=5, scoring='neg_mean_squared_error')
clf = clf.fit(X,y)
sc = -clf.cv_results_['mean_test_score']

# + colab={} colab_type="code" id="7xWskxA7LUEG" outputId="f9dc6a3a-1e55-4d21-e898-9ed06ae769fb"
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot(domain,sc) 
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.title(r'MSE al variare di $\alpha$ in Lasso')
plt.show()

# + colab={} colab_type="code" id="AGvgy--KLUEI" outputId="008111cc-83f4-414c-d260-b547e56e7f72"
min_index = np.argmin(sc)
print('Miglior valore per alpha: {0:.5f}. MSE={1:.3f}'.format(domain[min_index], sc[min_index]))

# + [markdown] colab_type="text" id="sRlwftSGLUEJ"
# Utilizzo di LassoCV, che ricerca il miglior valore di $\alpha$ valutando lo score su un insieme di possibili valori mediante cross validation. 

# + colab={} colab_type="code" id="-vPgG9HyLUEK"
domain=np.linspace(0,10,100)
p = Pipeline([('scaler', StandardScaler()),('regression', LassoCV(cv=10, alphas=domain))])
r = p.fit(X, y)
scores = np.mean(r.named_steps['regression'].mse_path_, axis=1)

# + colab={} colab_type="code" id="JTkdRiGhLUEL" outputId="6793e535-d5f3-4e6e-83f8-547d2148e0d3"
plt.figure(figsize=(16, 8))
plt.plot(r.named_steps['regression'].alphas_, scores)
plt.xlabel(r'$\alpha$')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()

# + colab={} colab_type="code" id="yCbKWZ7ULUEN" outputId="00646dab-68a3-4baf-d278-0371a8a008da"
best_alpha = r.named_steps['regression'].alpha_
print(r'Miglior valore di alpha: {0:.5f}'.format(best_alpha))
i, = np.where(r.named_steps['regression'].alphas_ == best_alpha)
print('MSE: {0:.5f}'.format(scores[i][0]))

# + colab={} colab_type="code" id="04lrHyLyLUEP" outputId="5b1748f3-1672-4754-b486-353da1be76a6"
r.named_steps['regression'].coef_

# + [markdown] colab_type="text" id="r9uv4O4ULUER"
# Valuta Lasso con il valore trovato per $\alpha$ sull'intero dataset

# + colab={} colab_type="code" id="2wbu_tUbLUER" outputId="8cdf778c-f32c-4d97-b44e-8f5f34ddf1bf"
p = Pipeline([('scaler', StandardScaler()),('regression', Lasso(alpha = best_alpha))])
scores = cross_val_score(estimator=p, X=X, y=y, cv=20, scoring='neg_mean_squared_error')
print('MSE: {0:.3f}'.format(-scores.mean()))

# + [markdown] colab_type="text" id="8HIFkbFOLUET"
# ### Ridge

# + [markdown] colab_type="text" id="pCRqGwfnLUEU"
# Ricerca su griglia di valori per alpha in Ridge

# + colab={} colab_type="code" id="HN3E3VxZLUEV"
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

# + colab={} colab_type="code" id="Kdi9vTo5LUEX" outputId="164f42a5-8c3e-47ea-8045-97049a923c07"
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot(scores[:,0], scores[:,1]) 
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.title(r'MSE al variare di $\alpha$ in Ridge')
plt.show()

# + colab={} colab_type="code" id="NQBRXCm_LUEb" outputId="5b805934-63fc-4781-b2f4-06259d5e0274"
min_index = np.argmin(scores[:,1])
best_alpha = scores[min_index,0]
print('Miglior valore per alpha: {0:.5f}. MSE={1:.3f}'.format(scores[min_index,0], scores[min_index,1]))

# + [markdown] colab_type="text" id="19v7tiI2LUEe"
# Applica sul dataset con il valore trovato per $\alpha$

# + colab={} colab_type="code" id="uhptAsMXLUEe" outputId="913d1d67-e752-4072-c687-36104da8965a"
p = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha = best_alpha))])
r = p.fit(X, y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('alpha: {0:.3f}, MSE: {1:.3f}'.format(best_alpha, -scores.mean()))

# + [markdown] colab_type="text" id="jHRPjs-QLUEg"
# Utilizzo di GridSearchCV

# + colab={} colab_type="code" id="PW0fB4x_LUEg"
domain = np.linspace(80,120,100)
param_grid = [{'regression__alpha': domain}]
p = Pipeline([('scaler', StandardScaler()),('regression', Ridge())])

clf = GridSearchCV(p, param_grid, cv=10, scoring='neg_mean_squared_error')
clf = clf.fit(X,y)
scores = -clf.cv_results_['mean_test_score']

# + colab={} colab_type="code" id="bxbCYRTuLUEi" outputId="9f7b9229-514b-46f0-e8ae-65040b55fde2"
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
plt.plot(domain,scores) 
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.title(r'MSE al variare di $\alpha$ in Ridge')
plt.show()

# + colab={} colab_type="code" id="eIQj0gPlLUEj" outputId="9cbbc2ec-6e1a-4d90-efe1-64d425d0c689"
min_index = np.argmin(scores)
print('Miglior valore per alpha: {0:.5f}. MSE={1:.3f}'.format(domain[min_index], scores[min_index]))

# + [markdown] colab_type="text" id="obkrxSSpLUEo"
# Applica sul dataset con il valore trovato per $\alpha$

# + colab={} colab_type="code" id="J2BspghOLUEo" outputId="5691e6b1-6853-4aa6-d842-3bede02a1d25"
p = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha = best_alpha))])
r = p.fit(X, y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('alpha: {0:.3f}, MSE: {1:.3f}'.format(best_alpha, -scores.mean()))

# + [markdown] colab_type="text" id="4Tqs9sR2LUEq"
# Utilizza RidgeCV, che ricerca il miglior valore di $\alpha$ valutando lo score su un insieme di possibili valori mediante cross validation

# + colab={} colab_type="code" id="iTVATZwZLUEq"
domain = np.linspace(0.1, 10, 100)
p = Pipeline([('scaler', StandardScaler()),('regression', RidgeCV(alphas=domain, store_cv_values = True))])
r = p.fit(X, y)
scores = np.mean(r.named_steps['regression'].cv_values_, axis=0)

# + colab={} colab_type="code" id="PfIO2uxuLUEr" outputId="82708b4c-0928-490d-d0b7-2f3db6d137ae"
plt.figure(figsize=(16, 8))
plt.plot(domain, scores)
plt.xlabel(r'$\alpha$')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()

# + colab={} colab_type="code" id="-sz6GwGfLUEt" outputId="2653a4ab-915b-486f-8af4-5482f9affeb5"
best_alpha = p.named_steps['regression'].alpha_
print(r'Miglior valore di alpha: {0:.6f}'.format(best_alpha))
i, = np.where(domain == best_alpha)
print('score: {0:.3f}'.format(scores[i][0]))

# + [markdown] colab_type="text" id="UNy1fvb2LUEu"
# Valuta Ridge con il valore trovato per  α
#   sull'intero dataset

# + colab={} colab_type="code" id="LQuzv8xPLUEv" outputId="4d426f89-9cd9-449c-9e92-8f30222b3d44"
p = Pipeline([('scaler', StandardScaler()),('regression', Ridge(alpha = best_alpha))])
r = p.fit(X, y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('alpha: {0:.3f}, MSE: {1:.3f}'.format(best_alpha, -scores.mean()))

# + colab={} colab_type="code" id="3nyzmtwNLUEw" outputId="72761f36-8137-44f8-f39c-cbc1eadfe60c"
r.named_steps['regression'].coef_

# + [markdown] colab_type="text" id="UDx5mHDrLUEx"
# ### Elastic net

# + [markdown] colab_type="text" id="k5kVlYdoLUEx"
# Ricerca su griglia 2d di valori per $\alpha$ e $\gamma$

# + colab={} colab_type="code" id="nFDgUlMCLUEy"
scores = []
for a in np.linspace(0,1,10):
    for l in np.linspace(0,1,10):
        p = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha=a, l1_ratio=l))])
        score = cross_val_score(estimator=p, X=X, y=y, cv=5, scoring='neg_mean_squared_error')
        scores.append([a,l,-score.mean()])

# + colab={} colab_type="code" id="0Y3gKi5VLUEz" outputId="fc164837-c661-4be8-d015-a2dc3964b0d7"
scores = np.array(scores)
min_index = np.argmin(scores[:,2])
best_alpha = scores[min_index, 0]
best_gamma = scores[min_index, 1]
print(r"Migliore coppia: alpha={0:.2f}, gamma={1:.2f}. MSE={2:.3f}".format(best_alpha,best_gamma, scores[min_index,2]))


# + colab={} colab_type="code" id="iMjKOfU-LUE0" outputId="5940ef2e-be53-4929-8005-a5ca828c126f"
p = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha = best_alpha, l1_ratio=best_gamma))])
r = p.fit(X, y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('alpha: {0:.3f}, gamma: {1:.3f}; MSE: {2:.3f}'.format(best_alpha, best_gamma, -scores.mean()))

# + [markdown] colab_type="text" id="y4Pi0frgLUE1"
# Utilizza GridsearchCV

# + colab={} colab_type="code" id="m46E7DZVLUE1"
param_grid = [{'regression__alpha': np.linspace(0,1,10), 'regression__l1_ratio': np.linspace(0,1,10)}]
p = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha=alpha, l1_ratio=gamma))])

clf = GridSearchCV(p, param_grid, cv=5, scoring='neg_mean_squared_error')
clf = clf.fit(X,y)
sc = -clf.cv_results_['mean_test_score']

# + colab={} colab_type="code" id="Z5TEh8u3LUE3" outputId="6afe7bca-88ee-47d8-c3af-69d67865e716"
best_alpha = clf.best_params_['regression__alpha']
best_gamma = clf.best_params_['regression__l1_ratio']
print(r"Migliore coppia: alpha={0:.2f}, gamma={1:.2f}. MSE={2:.3f}".format(best_alpha,
                                        best_gamma, -clf.best_score_))

# + colab={} colab_type="code" id="cKvNBVOvLUE5" outputId="0688aab9-5bde-46bb-a40c-97c7bd142dfd"
p = Pipeline([('scaler', StandardScaler()),('regression', ElasticNet(alpha = best_alpha, l1_ratio=best_gamma))])
r = p.fit(X, y)
scores = cross_val_score(estimator=p, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
print('alpha: {0:.3f}, gamma: {1:.3f}; MSE: {2:.3f}'.format(best_alpha, best_gamma, -scores.mean()))

# + colab={} colab_type="code" id="DznGMp-XLUE7"

