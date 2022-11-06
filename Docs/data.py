#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append('C:\\Users\\Peace\\Documents\\Project\\Final\\')
import features


# In[53]:


# %%
# Loading train and test data


# In[54]:


path = "C:\\Users\\Peace\\Documents\\Project\\Final\\"
train = pd.read_csv(path + 'data.csv')
test = pd.read_csv(path + 'test.csv')
print(train.info(memory_usage='deep')) # Using memory_usage parameter to include the size of object datatype columns too


# In[55]:


# %%
# Dropping id, qid1, qid2 columns - plays no role


# In[56]:


train = train.loc[:, ['question1', 'question2', 'is_duplicate']]
print(train.head())


# In[57]:


# %%
# Finding if there are any missing values in the data


# In[59]:


print(train.isnull().sum())
print(train[train.question2.isnull()]) # Index 105780 has a missing value, so let us remove this row from the data
# Dropping row with index 105780 and again checking if the file has any missing value in it


# In[61]:


print(train[train.question1.isnull()])


# In[62]:


train = train.drop(105780)
train = train.drop(363362)
train = train.drop(201841)
print(train.isnull().sum())


# In[63]:


# %%
# Making question columns to lowercase
train.question1 = train.question1.str.lower()
train.question2 = train.question2.str.lower()


# In[64]:


# %%
# Creating features of the train data and storing them in the x_train DataFrame


# In[65]:


x_train = pd.DataFrame()
x_train["X1"] = train.apply(features.word_match_share, axis=1, raw=True)
x_train["X2"] = train.apply(features.tfidf_word_match_share, axis=1, raw=True)
x_train["X3"] = pd.DataFrame(features.common_words_count(train))
X4, X5 = features.fuzz_sort_set(train)
x_train["X4"] = X4
x_train["X5"] = X5


# In[66]:


# %%
# Inferences from the first feature - Word Match Share 


# In[67]:


plt.hist(x_train.X1[train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
plt.hist(x_train.X1[train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Distribution of shared words', fontsize=20)
plt.ylabel('Ratio of shared words \n to total number of words', fontsize=15)
plt.show()


# In[68]:


# Inference from the above plot: Most of the questions (excluding the overlapping ones) 
# can be classified as duplicate and non-duplicate as confirmed from the graph.

# %%
# Inferences from the second feature - TF-IDF Word Match Share 


# In[69]:


plt.hist(x_train.X2[train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
plt.hist(x_train.X2[train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Distribution of shared weights', fontsize=20)
plt.ylabel('Ratio of shared weights \n to total weight', fontsize=15)
plt.show()


# In[70]:


# Even this feature is helpful in classifying either duplicate or not

# %%
# Inferences from the third feature - Common words count


# In[71]:


plt.hist(x_train.X3[train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
plt.hist(x_train.X3[train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Distribution of common words', fontsize=20)
plt.ylabel('Common words count', fontsize=15)
plt.show()


# In[72]:


# This feature may be slightly helpul as most of the data is overlapping

# %%
# Inferences from the third feature - Token Sort Ratio


# In[73]:


plt.hist(x_train.X4[train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
plt.hist(x_train.X4[train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Distribution of Token Sort Ratio', fontsize=20)
plt.ylabel('Token Sort Ratio', fontsize=15)
plt.show()


# In[74]:


# This one is quite helpful as a good amount of data is not overlapping

# %%
# Inferences from the third feature - Token Set Ratio


# In[75]:


plt.hist(x_train.X5[train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
plt.hist(x_train.X5[train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Distribution of Token Set Ratio', fontsize=20)
plt.ylabel('Token Set Ratio', fontsize=15)
plt.show()


# In[76]:


# Perfect, even this feature provides good insights.

# %%
# Analyzing these features based on ROC AUC metric


# In[77]:


print('          Original AUC:', roc_auc_score(train['is_duplicate'], x_train.X1))
print('             TFIDF AUC:', roc_auc_score(train['is_duplicate'], x_train.X2.fillna(0)))
print('Common Word Counts AUC:', roc_auc_score(train['is_duplicate'], x_train.X3))
print('  Token Sort Ratio AUC:', roc_auc_score(train['is_duplicate'], x_train.X4))
print('   Token Set Ratio AUC:', roc_auc_score(train['is_duplicate'], x_train.X5))


# In[78]:


# Inference - Feature importance is shown below:
# X1 > X5 > X4 > X3 > X2

# %%

# Splitting data in train and validation set for model building
# Considering 20% data as validation data


# In[79]:


X_t, X_v, y_t, y_v = train_test_split(x_train, train['is_duplicate'].values,
                                      test_size=0.2, random_state=42)


# In[80]:


# %%
# Model Building - XGBoost

# Setting parameters


# In[81]:


params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 2

d_train = xgb.DMatrix(X_t, label=y_t)
d_valid = xgb.DMatrix(X_v, label=y_v)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]


# In[82]:


# Training the model


# In[83]:


XGboostModel = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)


# In[84]:


# Saving the model parameters in an external file


# In[85]:


joblib.dump(XGboostModel, path + "XGboostModel.sav")


# In[86]:


# %%
# Model Building - LGBoost

# Setting parameters


# In[87]:


params = {}
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'    
params['eta'] = 0.02
params['max_depth'] = 2

lgtrain = lgb.Dataset(X_t, label=y_t)
lgval = lgb.Dataset(X_v, label=y_v)

LGboostModel = lgb.train(params, lgtrain, 400, valid_sets=[lgtrain, lgval], early_stopping_rounds=50, verbose_eval=10)


# In[88]:


# Saving the model parameters in an external file


# In[89]:


joblib.dump(LGboostModel, path + "LGboostModel.sav")


# In[90]:


# %%
# Model building - CatBoost


# In[91]:


cb_model = CatBoostRegressor(iterations=500, learning_rate=0.02, depth=10, 
                             eval_metric='RMSE', metric_period = 50)
CATboostModel = cb_model.fit(X_t, y_t, eval_set=(X_v, y_v), use_best_model=True, verbose=50)


# In[92]:


# Saving the model parameters in an external file


# In[93]:


joblib.dump(CATboostModel, path + "CATboostModel.sav")


# In[94]:


# %%
# Creating a test data


# In[95]:


test_sub = test.loc[:10,["question1", "question2"]].apply(lambda x: x.str.lower())


# In[96]:


x_test = pd.DataFrame()
x_test["X1"] = test_sub.apply(features.word_match_share, axis=1, raw=True)
x_test["X2"] = test_sub.apply(features.tfidf_word_match_share, axis=1, raw=True)
x_test["X3"] = pd.DataFrame(features.common_words_count(test_sub))
X4, X5 = features.fuzz_sort_set(test_sub)
x_test["X4"] = X4
x_test["X5"] = X5


# In[97]:


# %%
# Checking the models on few test data


# In[98]:


XG = joblib.load(path + "XGboostModel.sav")
LG = joblib.load(path + "LGboostModel.sav")
CAT = joblib.load(path + "CATboostModel.sav")


# In[99]:


# XGBoost prediction


# In[100]:


XG_pred = XG.predict(xgb.DMatrix(x_test)) * 100


# In[101]:


# LGBoost prediction


# In[102]:


LG_pred = np.expm1(LG.predict(x_test, num_iteration=LG.best_iteration)) * 100


# In[103]:


# CATBoost prediction


# In[104]:


CAT_pred = np.expm1(CAT.predict(x_test)) * 100


# In[105]:


# %%


# In[106]:


final_pred = test.loc[:10, ["question1", "question2", "is_duplicate"]]
final_pred["XGBoost Prediction in Percentage"] = XG_pred
final_pred["LGBoost Prediction in Percentage"] = LG_pred
final_pred["CATBoost Prediction in Percentage"] = CAT_pred
print(final_pred)


# In[ ]:





# In[ ]:




