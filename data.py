
# coding: utf-8

# In[81]:




from sklearn.feature_selection import SelectKBest, chi2, RFE, VarianceThreshold, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


# In[92]:


import pandas as pd
import numpy as np

import time


# In[83]:


mushroom = pd.read_csv('mushrooms.csv')
mushroom.drop('veil-type', inplace=True, axis=1)


# In[84]:


X = pd.get_dummies(mushroom, prefix=list(mushroom))
Y = X['class_p'].values
X.drop('class_e', inplace=True, axis=1)
X.drop('class_p', inplace=True, axis=1)
X = X.values


# In[85]:


num = 20
percent = .7
num_of_steps = 5


# In[94]:


# univariate feature extraction
start_time = time.time()
test = SelectKBest(score_func=chi2, k=num)
fit = test.fit(X, Y)
np.set_printoptions(precision=3)
X_univariate = fit.transform(X)
X_univariate.shape
print "univariate time elapsed: {} ".format(time.time() - start_time)


# In[98]:


# low variance
start_time = time.time()
sel = VarianceThreshold(threshold=(percent * (1 - percent)))
X_low_variance = sel.fit_transform(X)
X_low_variance.shape
print "low variance time elapsed: {} ".format(time.time() - start_time)


# In[97]:


# tree based feature selection
start_time = time.time()
clf = ExtraTreesClassifier(max_features=num)
clf.fit(X, Y)
model = SelectFromModel(estimator=clf, prefit=True)
X_tree = model.transform(X)
X_tree.shape
print "tree based time elapsed: {} ".format(time.time() - start_time)


# In[96]:


# PCA feature extraction
start_time = time.time()
pca = PCA(n_components=num)
fit = pca.fit(X)
# summarize components
# print "Explained Variance: %s" % fit.explained_variance_ratio_
X_PCA = fit.fit_transform(X, Y)
X_PCA.shape
print "PCA time elapsed: {} ".format(time.time() - start_time)


# In[95]:


# recursive feature elimination
start_time = time.time()
logreg = LogisticRegression()
rfe = RFE(logreg, num, step=num_of_steps)
fit = rfe.fit(X, Y)
# print("Selected Features: %s") % fit.support_
# print("Feature Ranking: %s") % fit.ranking_
X_RFE = fit.fit_transform(X, Y)
X_RFE.shape
print "RFE time elapsed: {} ".format(time.time() - start_time)


# In[ ]:


print ""

