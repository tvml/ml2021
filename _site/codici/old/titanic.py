#!/usr/bin/env python
# coding: utf-8

# ### Titanic disaster survivors
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew.  This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.  Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# We analyze here what sorts of people were likely to survive, in order to predict, for each passenger, her chances of surviving the shipwreck and to evaluate the overall prediction performance of the algorithms applied.

# In[1191]:


from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[1192]:


import matplotlib as mpl
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
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) 
cmap_big = cm.get_cmap('Spectral', 512)
cmap = mcolors.ListedColormap(cmap_big(np.linspace(0.7, 0.95, 256)))


# In[1193]:


import numpy as np
import pandas as pd 
import scipy.stats as stats
import seaborn.apionly as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve 
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression


# ### Data Handling
# #### Let's read our data in using pandas:

# In[1194]:


df = pd.read_csv("../dataset/titanic.csv") 


# Show an overview of our data: 

# In[1195]:


df.head(20)


# In[1196]:


df.shape


# In[1197]:


df.columns


# These are the meanings of each feature:
# 
# *Survival*: Survival(0 = No; 1 = Yes)  
# *Pclass*: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)  
# *Name*: Name  
# *Sex*: Sex (female, male)  
# *Age*: Age  
# *SibSp*: Number of Siblings/Spouses Aboard  
# *Parch*: Number of Parents/Children Aboard  
# *Ticket*: Ticket Number  
# *Fare*: Passenger Fare  
# *Cabin*: Cabin  
# *Embarked*: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[348]:


df.info()


# To have a better vision of the data we are going to display our feature with a countplot of seaborn. Show the counts of observations in each categorical bin using bars. The categorical features of our dataset are these are integer and object. We are going to separate our features into two lists: “categ” for the categorical features and “conti” for the continuous features. The “age” and the “fare” are the only two features that we can consider as continuous. In order to plot the distribution of the features with seaborn we are going to use distplot. According to the charts, there are no weird values (superior at 100) for “age” but we can see that the feature “fare” have a large scale and the most of value are between 0 and 100.

# In[535]:


df['TravelBuds'] = df["SibSp"]+df["Parch"]
df['Alone'] = np.where(df['TravelBuds']>0, 0, 1)
df.drop('TravelBuds', axis=1, inplace=True)


# In[350]:


categ =  [ 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Alone', 'Survived']
conti = ['Fare', 'Age']

#Distribution
fig = plt.figure(figsize=(16, 12))
for i in range (0,len(categ)):
    fig.add_subplot(3,3,i+1)
    sns.countplot(x=categ[i], data=df, alpha=.7) 

for col in conti:
    fig.add_subplot(3,3,i + 2)
    sns.distplot(df[col].dropna(), kde_kws={"lw": 2, "color":colors[8]}, 
                 hist_kws={"alpha": .5})
    i += 1
    
plt.show()


# In[373]:


fig = plt.figure(figsize=(16, 10))
i = 1
for col in categ:
    if col != 'Survived':
        fig.add_subplot(3,3,i)
        g = sns.countplot(x=col, data=df,hue='Survived', alpha=.7)
        plt.legend(loc=1) 
        i += 1

# Box plot survived x age
fig.add_subplot(3,3,7)
#sns.swarmplot(x="Survived", y="Age", hue="Sex", alpha=.7, data=df)
sns.distplot(df[df.Survived==0]['Age'].dropna(), bins = 20, kde_kws={"lw": 2}, 
                 hist_kws={"alpha": .4}, label='0')
sns.distplot(df[df.Survived==1]['Age'].dropna(), bins = 20, kde_kws={"lw": 2}, 
                 hist_kws={"alpha": .4}, label='1')
plt.legend()
fig.add_subplot(3,3,8)
sns.violinplot(x="Survived", y="Age", data=df, alpha=.7)

# fare and Survived
fig.add_subplot(3,3,9)
sns.violinplot(x="Survived", y="Fare", alpha=.7, data=df, saturation=.7)

plt.show()


# In[425]:


cm = df.drop(['PassengerId'], axis=1).corr()
mask = np.zeros_like(cm, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(16,8))
hm = sns.heatmap(cm,mask=mask,
                 annot=True,
                 fmt='.2f',
                 cmap = sns.diverging_palette(220, 10, as_cmap=True)
                , cbar_kws={"shrink": .5})
plt.tight_layout()
plt.show()


# ### Let us evaluate some conditional probabilities of surviving

# In[353]:


def cond_prob(feature):
    r = df[df[feature].notnull()][feature].unique()
    p = []
    for val in r:
        joint = df[(df[feature]==val) & (df['Survived']==1)].shape[0]
        pre = df[(df[feature]==val)].shape[0]
        post_1 = joint/pre
        p.append([val, post_1])
    return pd.DataFrame(p, columns=['value','prob'])


# In[429]:


def cond_prob_2(feature1, feature2):
    r1 = df[(df[feature1].notnull())&(df[feature2].notnull())][feature1].unique()
    r2 = df[(df[feature1].notnull())&(df[feature2].notnull())][feature2].unique()
    p = []
    for val1 in r1:
        for val2 in r2:
            joint = df[(df[feature1]==val1) & (df[feature2]==val2) & (df['Survived']==1)].shape[0]
            pre = df[(df[feature1]==val1) & (df[feature2]==val2)].shape[0]
            post_1 = joint/pre
            p.append([val1, val2, post_1])
    return pd.DataFrame(p, columns=[feature1,feature2,'prob'])


# In[435]:


cp = cond_prob_2('Sex','Pclass')
fig = plt.figure(figsize=(16, 6))
sns.barplot(x=cp.columns[0], y=cp.columns[2], hue=cp.columns[1], data=cp, alpha=.7)
plt.show()


# In[436]:


fig = plt.figure(figsize=(16, 10))
i = 1
for col in categ:
    if col!='Survived':
        fig.add_subplot(3,3,i)
        g = sns.barplot(x='value', y='prob', data=cond_prob(col), alpha=.7)
        plt.title(col, fontsize=12) 
        i += 1
plt.suptitle('Probability of surviving', y=1.05, fontsize=16)
plt.tight_layout()
plt.show()


# In[354]:


frq00, edges = np.histogram(df[df.Survived==1].Age, bins=range(0,90,10))
frq01, edges = np.histogram(df.Age, bins=range(0,90,10))
frq0, edges = np.histogram(df[df.Sex=='male'].Age, bins=range(0,90,10))
frq1, edges = np.histogram(df[(df.Survived==1) & (df.Sex=='male')].Age, bins=range(0,90,10))
frq2, edges = np.histogram(df[df.Sex=='female'].Age, bins=range(0,90,10))
frq3, edges = np.histogram(df[(df.Survived==1) & (df.Sex=='female')].Age, bins=range(0,90,10))


# In[439]:


fig = plt.figure(figsize=(16,4))
fig.add_subplot(1,2,1)
plt.bar(edges[:-1], frq00/frq01, width=np.diff(edges), ec="k", align="edge", color=colors[2], alpha=.7)
fig.add_subplot(1,2,2)
plt.bar(edges[:-1], frq3/frq2, width=np.diff(edges), ec="k", align="edge", color=colors[0], alpha=.5, label='female')
plt.bar(edges[:-1], frq1/frq0, width=np.diff(edges), ec="k", align="edge", color=colors[1], alpha=.5, label='male')
plt.legend()
plt.suptitle('Probability of surviving by age', fontsize=14)
plt.show()


# In[383]:


frq00, edges = np.histogram(df[df.Survived==1].Fare, bins=range(0,600,50))
frq01, edges = np.histogram(df.Fare, bins=range(0,600,50))
frq0, edges = np.histogram(df[df.Sex=='male'].Fare, bins=range(0,600,50))
frq1, edges = np.histogram(df[(df.Survived==1) & (df.Sex=='male')].Fare, bins=range(0,600,50))
frq2, edges = np.histogram(df[df.Sex=='female'].Fare, bins=range(0,600,50))
frq3, edges = np.histogram(df[(df.Survived==1) & (df.Sex=='female')].Fare, bins=range(0,600,50))


# In[440]:


fig = plt.figure(figsize=(16,4))
fig.add_subplot(1,2,1)
plt.bar(edges[:-1], frq00/frq01, width=np.diff(edges), ec="k", align="edge", color=colors[2], alpha=.7)
fig.add_subplot(1,2,2)
plt.bar(edges[:-1], frq3/frq2, width=np.diff(edges), ec="k", align="edge", color=colors[0], alpha=.5, label='female')
plt.bar(edges[:-1], frq1/frq0, width=np.diff(edges), ec="k", align="edge", color=colors[1], alpha=.5, label='male')
plt.legend()
plt.suptitle('Probability of surviving by fare', fontsize=14)
plt.show()


# In[536]:


df.columns


# ## Features tweaking

# First of all, let us get rid of features which appear clearly unrelated to surviving probability, such as *PassengerId*, *Name*, *Ticket*

# In[537]:


df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1) 


# In[538]:


df.columns


# #### Missing values

# In[539]:


df.columns[df.isna().any()].tolist()


# *Age* has a limited number of missing values

# In[540]:


print('Number of null values: {0:d}'.format(sum(pd.isnull(df['Age']))))
print('Ratio of null values: {0:5.3f}'.format(sum(pd.isnull(df['Age']))/df.shape[0]))


# We can deal with the problem by inserting a suitable value to fill missing locations. Let us look at the distribution of ages

# In[448]:


plt.figure(figsize=(16,4))
sns.distplot(df[df['Age'].notnull()]['Age'],hist=True,bins=20, kde_kws={'color':colors[8]})
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# *Age* is (right) skewed: using the mean might give us biased results by filling in ages that are older than desired. To deal with this, we use the median to impute the missing values.

# In[541]:


df['Age'] = df['Age'].fillna(df['Age'].median(skipna=True))


# *Cabin* has a larger amount of missing values. 

# In[449]:


print('Number of null values: {0:d}'.format(sum(pd.isnull(df['Cabin']))))
print('Ratio of null values: {0:5.3f}'.format(sum(pd.isnull(df['Cabin']))/df.shape[0]))


# Moreover, it do not seem to add much value to the analysis: let us drop it from the dataframe.

# In[542]:


df = df.drop(['Cabin'], axis=1) 
df.shape


# *Embarked* has just a few missing values

# In[451]:


print('Number of null values: {0:d}'.format(sum(pd.isnull(df['Embarked']))))
print('Ratio of null values: {0:5.3f}'.format(sum(pd.isnull(df['Embarked']))/df.shape[0]))


# Being it a categorical features, we fill missing values with the most frequent class

# In[543]:


df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().argmax())


# In[544]:


df.info()


# ## One-hot encoding of categorical variables
# This apply to *Sex*, *Pclass* and *Embarked*

# In[545]:


df = pd.get_dummies(df,drop_first=True,columns=['Sex', 'Embarked', 'Pclass'])


# In[546]:


df.head()


# ## Class balance
# Let us check that target classes are balanced

# In[547]:


print('Number of positive items: {0:d}'.format(df[df.Survived==1].shape[0]))
print('Number of negative items: {0:d}'.format(df[df.Survived==0].shape[0]))
print('Fraction of positive items: {0:3.2f}'.format(df[df.Survived==1].shape[0]/df.shape[0]))


# There is a certain unbalancement towards negative items. Let us consider it acceptable, for now.

# Let us derive the feature matrix and the target vector.

# In[548]:


X = df.drop(['Survived'], axis=1)
y = df['Survived']


# In[549]:


scaler = StandardScaler()
scaler = scaler.fit(X)
X_s = pd.DataFrame(scaler.transform(X), columns=X.columns)


# ## Feature selection

# In[550]:


forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

forest.fit(X_s, y)
importances = forest.feature_importances_


# In[551]:


ff = np.array([e.feature_importances_ for e in forest.estimators_])
dd = pd.DataFrame(ff, columns=X.columns)


# In[552]:


fig = plt.figure(figsize=(16, 6))
sns.barplot(data=dd, ci="sd", alpha=.7)
plt.title('Feature relevance for classification')
plt.show()


# In[553]:


mi = mutual_info_classif(X_s, y)
dmi = pd.DataFrame(mi, index=X.columns, columns=['mi']).sort_values(by='mi', ascending=False)
dmi


# In[554]:


fc = SelectKBest(f_classif, k='all').fit(X_s, y)
dfc = pd.DataFrame(np.array([fc.scores_, fc.pvalues_]).T, index=X.columns, columns=['score','pval']).sort_values(by='score', ascending=False)
dfc


# It seems that 5 features (*Sex_male*, *Pclass_3*, *Fare*, *Alone*, *Age*) are the ones providing most information needed for  prediction. Let us define a reduced matrix containing only values from those features and a function which selects such features from a dataframe

# In[794]:


X_sel = X_s[['Sex_male', 'Pclass_3', 'Fare', 'Alone', 'Age']]
cols = [X.columns.get_loc(c) for c in ['Sex_male', 'Pclass_3', 'Fare', 'Alone', 'Age']]


# In[797]:


def select_columns(X):
    return np.take(X,cols, axis=1)


# We can reduce the dimensionality of the problem also by feature extraction for example by applying PCA

# In[559]:


pca = PCA(n_components=X_s.shape[1])
pca = pca.fit(X_s)


# In[560]:


evr = pca.explained_variance_ratio_


# In[561]:


fig = plt.figure(figsize=(16,6))
plt.bar(range(evr.shape[0]), evr.cumsum(), alpha=.7)
plt.xticks(range(evr.shape[0]))
plt.yticks(np.linspace(0,1,11))
plt.show()


# In[563]:


X_pca = PCA(n_components=8).fit_transform(X_s)


# In[565]:


X_pca.shape


# ## Prediction

# ### Naive Bayes
# Classification is performed by creating a classifier and insert it in a pipeline, after a scaler

# In[910]:


clf = Pipeline([('scaler', StandardScaler()),('classifier', GaussianNB())])


# The classifier is trained by submitting the training set X,y to it. Let us first refer to all features

# In[911]:


clf = clf.fit(X,y)


# The classifier coefficients are now instantiated according to the submitted training set.  
# The classifier can now be applied to prediction. Let us first apply it to the traning set itself (indeed, only to the feature matrix X)

# In[912]:


p = clf.predict(X)


# ## How well did the classifier behave?

# We may use several measures here. The ones we consider refer to the following values:
# 
# *True positive*: the number of survived passengers (i.e. belonging to class 1) correctly classified  
# *True negative*: the number of died passengers (i.e. belonging to class 0) correctly classified  
# *False positive*: the number of died passengers (i.e. belonging to class 0) classified as survived  
# *False negative*: the number of survived passengers (i.e. belonging to class 1) classified as died
# 
# Such values can be reported in a *confusion matrix*. By definition a confusion matrix $C$ is such that $C(i, j)$ is equal to the number of observations known to be in group $i$ but predicted to be in group $j$. That is, rows correspond to classes, columns to predictions.

# In[836]:


print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in [['TN', 'FP'],['FN','TP']]]))


# In[913]:


cm = confusion_matrix(p, y)


# In[914]:


plt.figure(figsize=(14,3))
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='d',annot_kws={'size': 14},
                 cmap = sns.color_palette("PuBu", 10))
plt.tight_layout()
plt.show()


# By definition a confusion matrix C is such that C(i, j) is equal to the number of observations known to be in group i but predicted to be in group j. That is, rows correspond to classes, columns to predictions. So, we have here

# In[915]:


print('{0:4d} true positives'.format(cm[1,1]))
print('{0:4d} false negatives'.format(cm[1,0]))
print('{0:4d} false positives'.format(cm[0,1]))
print('{0:4d} true negatives'.format(cm[0,0]))


# *Precision* is defined as the ratio of elements predicted as positive which are indeed positive: P=TP/(TP+FP). In this case, the ratio of survived passenger correctly classified wrt the total number of passengers classified as survived
# 
# *Recall* is defined as the ratio of positive elements which are predicted as positive: P=TP/(TP+FN). In this case, the ratio of survived passenger correctly classified wrt the total number of survived passengers
# 
# *F-score* is the harmonic mean of precision and recall $fscore = (precision^{-1}+recall^{-1})^{-1}$
# 
# *Accuracy* is defined as the ratio of correctly classified elements wrt to the overal number of elements

# In[916]:


print('Precision = {0:5.4f}'.format(precision_score(p,y)))
print('Recall = {0:5.4f}'.format(recall_score(p,y)))
print('F-score = {0:5.4f}'.format(f1_score(p,y)))
print('Accuracy = {0:5.4f}'.format(accuracy_score(p,y)))


# We may however be interested in precision, recall and f-score for both classes (survived and perished)

# In[917]:


m = precision_recall_fscore_support(p,y)


# In[918]:


print('Precision class 0 = {0:5.4f}'.format(m[0][0]))
print('Precision class 1 = {0:5.4f}'.format(m[0][1]))
print('Recall class 0 = {0:5.4f}'.format(m[1][0]))
print('Recall class 1 = {0:5.4f}'.format(m[1][1]))
print('F-score class 0 = {0:5.4f}'.format(m[2][0]))
print('F-score class 1 = {0:5.4f}'.format(m[2][1]))
print('Accuracy = {0:5.4f}'.format(accuracy_score(p,y)))


# The model also returns measures of the confidence of passenger predictions, in terms of probabilities. 
# 
# *class probabilities*: the estimated probability that a passenger is survived

# In[1016]:


y_prob = clf.predict_proba(X)
c = list(y.apply(lambda x: colors[1] if x==1 else colors[2]))
plt.figure(figsize=(18,10))
plt.scatter(range(y_prob.shape[0]), y_prob[:,1], color=c, s = 20, marker='o', alpha=.7)
plt.plot(range(y_prob.shape[0]), y, color=colors[0], ms = 4, marker='o', linestyle=' ', alpha=.7)
plt.axhline(y=.5, xmin=0, xmax=1, linewidth=2, linestyle='dashed', color=colors[9])
plt.xlim(1,y_prob.shape[0])
plt.title('Survived: Blue: \nPerished: Red');
plt.show()


# A ROC (Receiver Operating Characteristic) curve represents, for any value of the threshold applied to the probabilities for classification, the ratio $\frac{FP}{N}=\frac{FP}{TN+FP}$ of negative elements incorrectly classified as positive (*False positive rate*) and the ratio $\frac{TP}{P}=\frac{TP}{TP+FN}$ of positive elements correctly classified as positive (*True positive rate*).
# 
# The ideal point is (0,1): no negative elements misclassified, all positive elements well classified.

# In[919]:


fpr, tpr, thresholds = roc_curve(y, clf.predict_proba(X)[:,1], pos_label=1)


# In[920]:


plt.figure(figsize=(16,8))
plt.plot(fpr, tpr, color=colors[0], linewidth=1,label='ROC curve (area = %0.2f)' %  auc(fpr, tpr))
plt.fill_between(fpr, 0, tpr, alpha=0.2)
plt.plot([0, 1], [0, 1], color=colors[1], linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel(r'True Positive Rate')
plt.xlabel(r'False Positive Rate')
plt.title('Receiver operating characteristic', fontsize=14)
plt.legend(loc="lower right")
plt.show()


# The Area under curve (AUC) provides a measure of the quality of the classifier. More precisely, it corresponds to the probability that a classifier ranks a random positive item higher than a random negative one (that is assigns it a higher probability of being positive).

# In[921]:


print('AUC = {0:0.4f}'.format(auc(fpr, tpr)))


# In[922]:


roc_auc_score(y, clf.predict_proba(X)[:,1])


# In[923]:


optimal_idx = np.argmax(tpr-fpr)
optimal_threshold = thresholds[optimal_idx]


# In[924]:


print('Optimal threshold: {0:3.3f}'.format(optimal_threshold))


# In[925]:


pred_proba_df = pd.DataFrame(clf.predict_proba(X)[:,1])
p = pred_proba_df.applymap(lambda x: 1 if x>optimal_threshold else 0)


# In[926]:


cm = confusion_matrix(y,p)


# In[927]:


plt.figure(figsize=(14,3))
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='d',annot_kws={'size': 14},
                 cmap = sns.color_palette("PuBu", 10))
plt.tight_layout()
plt.show()


# In[928]:


m = precision_recall_fscore_support(p,y)


# In[929]:


print('Precision class 0 = {0:5.4f}'.format(m[0][0]))
print('Precision class 1 = {0:5.4f}'.format(m[0][1]))
print('Recall class 0 = {0:5.4f}'.format(m[1][0]))
print('Recall class 1 = {0:5.4f}'.format(m[1][1]))
print('F-score class 0 = {0:5.4f}'.format(m[2][0]))
print('F-score class 1 = {0:5.4f}'.format(m[2][1]))
print('Accuracy = {0:5.4f}'.format(accuracy_score(p,y)))


# A better evaluation can be obtained by using a training set - test set pair

# In[930]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[931]:


clf = Pipeline([('scaler', StandardScaler()),('classifier', GaussianNB())]).fit(X_train, y_train)


# Perform predictions on both training and test set

# In[932]:


p_train = clf.predict(X_train)
p_test = clf.predict(X_test)


# Compute measures in both cases

# In[933]:


cm_train = confusion_matrix(y_train,p_train)
cm_test = confusion_matrix(y_test,p_test)


# In[934]:


plt.figure(figsize=(14,3))
plt.subplot(1,2,1)
hm = sns.heatmap(cm_train,cbar=True,annot=True,square=True,fmt='d',annot_kws={'size': 12},cmap = sns.color_palette("PuBu", 10))
plt.title('Training set', fontsize=14)
plt.subplot(1,2,2)
hm = sns.heatmap(cm_test,cbar=True,annot=True,square=True,fmt='d',annot_kws={'size': 12},cmap = sns.color_palette("OrRd", 10))
plt.title('Test set', fontsize=14)
plt.tight_layout()
plt.show()


# In[935]:


m_train = precision_recall_fscore_support(y_train,p_train)
m_test = precision_recall_fscore_support(y_test,p_test)


# In[936]:


print('TRAINING SET')
print('Precision class 0 = {0:5.4f}'.format(m_train[0][0]))
print('Precision class 1 = {0:5.4f}'.format(m_train[0][1]))
print('Recall class 0 = {0:5.4f}'.format(m_train[1][0]))
print('Recall class 1 = {0:5.4f}'.format(m_train[1][1]))
print('F-score class 0 = {0:5.4f}'.format(m_train[2][0]))
print('F-score class 1 = {0:5.4f}'.format(m_train[2][1]))
print('Accuracy = {0:5.4f}'.format(accuracy_score(p_train,y_train)))


# In[937]:


print('TEST SET')
print('Precision class 0 = {0:5.4f}'.format(m_test[0][0]))
print('Precision class 1 = {0:5.4f}'.format(m_test[0][1]))
print('Recall class 0 = {0:5.4f}'.format(m_test[1][0]))
print('Recall class 1 = {0:5.4f}'.format(m_test[1][1]))
print('F-score class 0 = {0:5.4f}'.format(m_test[2][0]))
print('F-score class 1 = {0:5.4f}'.format(m_test[2][1]))
print('Accuracy = {0:5.4f}'.format(accuracy_score(p_test,y_test)))


# In[939]:


y_prob = clf.predict_proba(X_train)
print('AUC:{0:3.3f}'.format(roc_auc_score(y_train, y_prob[:,1])))


# A more stable evaluation can be obtained by cross validation (here 7-fold)

# In[940]:


scores = cross_validate(estimator=clf, X=X, y=y, cv=7, scoring=('precision', 'recall', 'accuracy', 'f1', 'roc_auc'),
                        return_train_score=True)


# In[941]:


print('AUC:{0:3.3f}'.format(scores['test_roc_auc'].mean()))


# Let us now consider the reduced dataset with selected  features

# In[942]:


clf = Pipeline([('scaler', StandardScaler()),('features', FunctionTransformer(select_columns)),
                ('classifier', GaussianNB())])


# In[943]:


scores = cross_validate(estimator=clf, X=X, y=y, cv=7, scoring=('precision', 'recall', 'accuracy', 'f1', 'roc_auc'),
                        return_train_score=True)


# In[944]:


print('AUC:{0:3.3f}'.format(scores['test_roc_auc'].mean()))


# And also the one reduced by PCA, with 8 components

# In[1017]:


clf = Pipeline([('scaler', StandardScaler()),('features', PCA(n_components=8)),
                ('classifier', GaussianNB())])


# In[1018]:


scores = cross_validate(estimator=clf, X=X, y=y, cv=7, scoring=('precision', 'recall', 'accuracy', 'f1', 'roc_auc'),
                        return_train_score=True)


# In[1019]:


print('AUC:{0:3.3f}'.format(scores['test_roc_auc'].mean()))


# In[1020]:


clf = clf.fit(X,y)


# In[1021]:


fpr, tpr, thresholds = roc_curve(y, clf.predict_proba(X)[:,1], pos_label=1)
optimal_idx = np.argmax(tpr-fpr)
optimal_threshold = thresholds[optimal_idx]
pred_proba_df = pd.DataFrame(clf.predict_proba(X)[:,1])
p = pred_proba_df.applymap(lambda x: 1 if x>optimal_threshold else 0)
print('Optimal threshold: {0:3.3f}'.format(optimal_threshold))


# In[1023]:


# Plot Predictions Vs Actual
y_prob = clf.predict_proba(X)
c = list(y.apply(lambda x: colors[1] if x==1 else colors[2]))
plt.figure(figsize=(18,10))
plt.scatter(range(y_prob.shape[0]), y_prob[:,1], color=c, s = 20, marker='o', alpha=.7)
plt.plot(range(y_prob.shape[0]), y, color=colors[0], ms = 4, marker='o', linestyle=' ', alpha=.7)
plt.axhline(y=optimal_threshold, xmin=0, xmax=1, linewidth=2, linestyle='dashed', color=colors[9])
plt.xlim(1,y_prob.shape[0])
plt.title('Survived: Blue: \nPerished: Red');
plt.show()


# In[950]:


cm = confusion_matrix(y,p)
plt.figure(figsize=(14,3))
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='d',annot_kws={'size': 14},
                 cmap = sns.color_palette("PuBu", 10))
plt.tight_layout()
plt.show()


# In[951]:


m = precision_recall_fscore_support(p,y)
print('Precision class 0 = {0:5.4f}'.format(m[0][0]))
print('Precision class 1 = {0:5.4f}'.format(m[0][1]))
print('Recall class 0 = {0:5.4f}'.format(m[1][0]))
print('Recall class 1 = {0:5.4f}'.format(m[1][1]))
print('F-score class 0 = {0:5.4f}'.format(m[2][0]))
print('F-score class 1 = {0:5.4f}'.format(m[2][1]))
print('Accuracy = {0:5.4f}'.format(accuracy_score(p,y)))


# ### Logistic regression

# In[1024]:


clf = Pipeline([('scaler', StandardScaler()),('classifier', LogisticRegression())])


# In[1025]:


scores = cross_validate(estimator=clf, X=X, y=y, cv=7, scoring=('precision', 'recall', 'accuracy', 'f1', 'roc_auc'),
                        return_train_score=True)


# In[1026]:


print('AUC:{0:3.3f}'.format(scores['test_roc_auc'].mean()))


# In[1027]:


print('Accuracy:{0:3.3f}'.format(scores['test_accuracy'].mean()))


# In[1028]:


clf = clf.fit(X,y)


# In[1029]:


fpr, tpr, thresholds = roc_curve(y, clf.predict_proba(X)[:,1], pos_label=1)
optimal_idx = np.argmax(tpr-fpr)
optimal_threshold = thresholds[optimal_idx]
pred_proba_df = pd.DataFrame(clf.predict_proba(X)[:,1])
p = pred_proba_df.applymap(lambda x: 1 if x>optimal_threshold else 0)
print('Optimal threshold: {0:3.3f}'.format(optimal_threshold))


# In[1030]:


# Plot Predictions Vs Actual
y_prob = clf.predict_proba(X)
c = list(y.apply(lambda x: colors[1] if x==1 else colors[2]))
plt.figure(figsize=(18,10))
plt.scatter(range(y_prob.shape[0]), y_prob[:,1], color=c, s = 20, marker='o', alpha=.7)
plt.plot(range(y_prob.shape[0]), y, color=colors[0], ms = 4, marker='o', linestyle=' ', alpha=.7)
plt.axhline(y=optimal_threshold, xmin=0, xmax=1, linewidth=2, linestyle='dashed', color=colors[9])
plt.xlim(1,y_prob.shape[0])
plt.title('Survived: Blue: \nPerished: Red');
plt.show()


# In[1031]:


cm = confusion_matrix(y,p)
plt.figure(figsize=(14,3))
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='d',annot_kws={'size': 14},
                 cmap = sns.color_palette("PuBu", 10))
plt.tight_layout()
plt.show()


# In[1032]:


m = precision_recall_fscore_support(p,y)
print('Precision class 0 = {0:5.4f}'.format(m[0][0]))
print('Precision class 1 = {0:5.4f}'.format(m[0][1]))
print('Recall class 0 = {0:5.4f}'.format(m[1][0]))
print('Recall class 1 = {0:5.4f}'.format(m[1][1]))
print('F-score class 0 = {0:5.4f}'.format(m[2][0]))
print('F-score class 1 = {0:5.4f}'.format(m[2][1]))
print('Accuracy = {0:5.4f}'.format(accuracy_score(p,y)))


# ### Regularization
# Let us introduce a regularization component, with an associated coefficient $C$

# In[1135]:


domain = np.linspace(0.1,10,100)
param_grid = [{'classifier__C': domain, 'classifier__penalty': ['l1','l2']}]
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
p = Pipeline([('scaler', StandardScaler()),('classifier', LogisticRegression())])

clf = GridSearchCV(p, param_grid, cv=5, scoring=scoring, refit='AUC', return_train_score=True)
clf = clf.fit(X,y)


# In[1136]:


results = clf.cv_results_


# In[1137]:


df_results = pd.DataFrame(results)


# In[1186]:


plt.figure(figsize=(16, 8))
dd = df_results[df_results.param_classifier__penalty=='l1']
X_axis = np.array(dd['param_classifier__C'])

mn = 10000
mx = 0
for scorer, color in zip(sorted(scoring), [colors[0], colors[1]]):
    for sample, style in (('train', '--'), ('test', '-')):
        mean = dd['mean_%s_%s' % (sample, scorer)]
        std = dd['std_%s_%s' % (sample, scorer)]
        #plt.fill_between(X_axis, mean-std, mean+std,alpha=0.3 if sample == 'test' else 0, color=color)
        plt.plot(X_axis, mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))
        mn = min(mn, min(mean-std))
        mx = max(mx, max(mean+std))
    scores = np.array(dd['mean_test_%s'% scorer])
    best_index = scores.argmax()
    best_score = scores[best_index]
    plt.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    plt.annotate("%0.3f" % best_score, (X_axis[best_index], best_score + 0.002))

plt.xlabel("$C$")
plt.ylabel("Score")

plt.xlim(0.1, 10)
plt.ylim(mn, mx)
plt.legend()
plt.title("Scores for train and test set, L1 penalty", fontsize=14)
plt.show()


# In[1187]:


domain = np.linspace(0.01,1,100)
param_grid = [{'classifier__C': domain, 'classifier__penalty': ['l1','l2']}]
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
p = Pipeline([('scaler', StandardScaler()),('classifier', LogisticRegression())])

clf = GridSearchCV(p, param_grid, cv=5, scoring=scoring, refit='AUC', return_train_score=True)
clf = clf.fit(X,y)

results = clf.cv_results_


# In[1189]:


df_results = pd.DataFrame(results)
plt.figure(figsize=(16, 8))
dd = df_results[df_results.param_classifier__penalty=='l1']
X_axis = np.array(dd['param_classifier__C'])

mn = 10000
mx = 0
for scorer, color in zip(sorted(scoring), [colors[0], colors[1]]):
    for sample, style in (('train', '--'), ('test', '-')):
        mean = dd['mean_%s_%s' % (sample, scorer)]
        std = dd['std_%s_%s' % (sample, scorer)]
        plt.plot(X_axis, mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))
        mn = min(mn, min(mean-std))
        mx = max(mx, max(mean+std))
    scores = np.array(dd['mean_test_%s'% scorer])
    best_index = scores.argmax()
    best_score = scores[best_index]
    plt.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    plt.annotate("%0.3f" % best_score, (X_axis[best_index], best_score + 0.002))

plt.xlabel("$C$")
plt.ylabel("Score")

plt.xlim(0.01, 1)
plt.ylim(mn, mx)
plt.legend()
plt.title("Scores for train and test set, L1 penalty", fontsize=14)
plt.show()


# In[1132]:


domain = np.linspace(0.05,0.25,100)
param_grid = [{'classifier__C': domain, 'classifier__penalty': ['l1','l2']}]
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
p = Pipeline([('scaler', StandardScaler()),('classifier', LogisticRegression())])

clf = GridSearchCV(p, param_grid, cv=5, scoring=scoring, refit='AUC', return_train_score=True)
clf = clf.fit(X,y)

results = clf.cv_results_


# In[1190]:


df_results = pd.DataFrame(results)
plt.figure(figsize=(16, 8))
dd = df_results[df_results.param_classifier__penalty=='l1']
X_axis = np.array(dd['param_classifier__C'])

mn = 10000
mx = 0
for scorer, color in zip(sorted(scoring), [colors[0], colors[1]]):
    for sample, style in (('train', '--'), ('test', '-')):
        mean = dd['mean_%s_%s' % (sample, scorer)]
        std = dd['std_%s_%s' % (sample, scorer)]
        plt.plot(X_axis, mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))
        mn = min(mn, min(mean-std))
        mx = max(mx, max(mean+std))
    scores = np.array(dd['mean_test_%s'% scorer])
    best_index = scores.argmax()
    best_score = scores[best_index]
    plt.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    plt.annotate("%0.3f" % best_score, (X_axis[best_index], best_score + 0.002))

plt.xlabel("$C$")
plt.ylabel("Score")

plt.xlim(0.05, 0.25)
plt.ylim(mn, mx)
plt.legend()
plt.title("Scores for train and test set, L1 penalty", fontsize=14)
plt.show()


# In[ ]:




