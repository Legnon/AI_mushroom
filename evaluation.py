
# coding: utf-8

# In[131]:




from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
import time


# In[132]:


from data import Y, X, X_univariate, X_low_variance, X_PCA, X_RFE, X_tree


# In[ ]:


def do_learning(algorithm, name):
    global nparray
    print "------- {} -------".format(name)
    start_time = time.time()
    scores = [name]
    for key, value in X_vars.items():
        score = cross_val_score(algorithm, value, Y, cv=folding, scoring='mean_squared_error')
        print "mean value of {}: {}".format(key, np.sqrt(-score.mean()))
        scores += [-score.mean()]
    print "time elapsed: {} ".format(time.time() - start_time)
    print ""
    nparray = np.vstack([nparray, scores])


# In[133]:


X_vars = {
    'X': X,
    'X_univariate': X_univariate,
    'X_low_variance': X_low_variance,
    'X_PCA': X_PCA,
    'X_RFE': X_RFE, 
    'X_tree': X_tree
}


# In[134]:


linreg = LinearRegression()
logreg = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=1)
dectree = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
ranforest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
extree = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
adaboost = AdaBoostClassifier(n_estimators=10)

scores = ['']
for k, v in X_vars.items():
    scores += [k]

nparray = np.array([scores])


# In[135]:


selectline = """
[1] Linear Regression
[2] Logistic Regression
[3] K Nearest Neighbors
[4] Random Forest Classifier
[5] Extra Tree Classifier
[6] Decision Tree Classifier
[7] Ada Boost Classifier
[8] ALL
"""
select = str(input(selectline))

folding = input('Folding? ')
print ""


# In[136]:


if "1" in select or "8" in select:
    do_learning(linreg, "Linear Regression")


# In[137]:


if "2" in select or "8" in select:
    do_learning(logreg, "Logistic Regression")


# In[138]:


if "3" in select or "8" in select:
    do_learning(knn, "KNN")


# In[139]:


if "4" in select or "8" in select:
    do_learning(ranforest, "Random Forest")


# In[140]:


if "5" in select or "8" in select:
    do_learning(extree, "Extra Tree Classifier")


# In[141]:


if "6" in select or "8" in select:
    do_learning(dectree, "Decision Tree Classifier")


# In[142]:


if "7" in select or "8" in select:
    do_learning(adaboost, "AdaBoost Classifier")


# In[147]:


nparray
df = pd.DataFrame(nparray)
df.to_csv('./result_cv{}.csv'.format(folding), header=None, index=None)

