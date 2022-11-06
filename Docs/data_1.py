#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# %%
# Loading train and test data


# In[3]:


path = "C:\\Users\\Peace\\Documents\\Project\\Final\\"
data = pd.read_csv(path + 'data.csv')
data


# In[4]:


train = data.iloc[:363861]
test = data.iloc[363861:]
print(train.info(memory_usage='deep')) # Using memory_usage parameter to include the size of object datatype columns too


# In[5]:


# %%
# Dropping id, qid1, qid2 columns - plays no role


# In[6]:


train = train.loc[:, ['question1', 'question2', 'is_duplicate']]
print(train.head())


# In[8]:


# %%
# Finding if there are any missing values in the data


# In[7]:


print(train.isnull().sum())


# In[ ]:





# In[8]:


train.dropna(subset=['question1', 'question2'], inplace=True)


# In[9]:


print(train.isnull().sum())


# In[ ]:





# In[10]:


# %%
# Making question columns to lowercase
train.question1 = train.question1.str.lower()
train.question2 = train.question2.str.lower()


# In[11]:


# %%
# Creating features of the train data and storing them in the x_train DataFrame


# In[12]:


x_train = pd.DataFrame()
x_train["X1"] = train.apply(features.word_match_share, axis=1, raw=True)
x_train["X2"] = train.apply(features.tfidf_word_match_share, axis=1, raw=True)
x_train["X3"] = pd.DataFrame(features.common_words_count(train))
X4, X5 = features.fuzz_sort_set(train)
x_train["X4"] = X4
x_train["X5"] = X5


# In[13]:


# %%
# Inferences from the first feature - Word Match Share 


# In[14]:


plt.hist(x_train.X1[train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
plt.hist(x_train.X1[train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Distribution of shared words', fontsize=20)
plt.ylabel('Ratio of shared words \n to total number of words', fontsize=15)
plt.show()


# In[15]:


# Inference from the above plot: Most of the questions (excluding the overlapping ones) 
# can be classified as duplicate and non-duplicate as confirmed from the graph.

# %%
# Inferences from the second feature - TF-IDF Word Match Share 


# In[16]:


plt.hist(x_train.X2[train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
plt.hist(x_train.X2[train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Distribution of shared weights', fontsize=20)
plt.ylabel('Ratio of shared weights \n to total weight', fontsize=15)
plt.show()


# In[17]:


# Even this feature is helpful in classifying either duplicate or not

# %%
# Inferences from the third feature - Common words count


# In[18]:


plt.hist(x_train.X3[train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
plt.hist(x_train.X3[train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Distribution of common words', fontsize=20)
plt.ylabel('Common words count', fontsize=15)
plt.show()


# In[19]:


# This feature may be slightly helpul as most of the data is overlapping

# %%
# Inferences from the third feature - Token Sort Ratio


# In[20]:


plt.hist(x_train.X4[train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
plt.hist(x_train.X4[train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Distribution of Token Sort Ratio', fontsize=20)
plt.ylabel('Token Sort Ratio', fontsize=15)
plt.show()


# In[21]:


# This one is quite helpful as a good amount of data is not overlapping

# %%
# Inferences from the third feature - Token Set Ratio


# In[22]:


plt.hist(x_train.X5[train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
plt.hist(x_train.X5[train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Distribution of Token Set Ratio', fontsize=20)
plt.ylabel('Token Set Ratio', fontsize=15)
plt.show()


# In[23]:


# Perfect, even this feature provides good insights.

# %%
# Analyzing these features based on ROC AUC metric


# In[24]:


print('          Original AUC:', roc_auc_score(train['is_duplicate'], x_train.X1))
print('             TFIDF AUC:', roc_auc_score(train['is_duplicate'], x_train.X2.fillna(0)))
print('Common Word Counts AUC:', roc_auc_score(train['is_duplicate'], x_train.X3))
print('  Token Sort Ratio AUC:', roc_auc_score(train['is_duplicate'], x_train.X4))
print('   Token Set Ratio AUC:', roc_auc_score(train['is_duplicate'], x_train.X5))


# In[25]:


# Inference - Feature importance is shown below:
# X1 > X5 > X4 > X3 > X2

# %%

# Splitting data in train and validation set for model building
# Considering 20% data as validation data


# In[26]:


X_t, X_v, y_t, y_v = train_test_split(x_train, train['is_duplicate'].values,
                                      test_size=0.2, random_state=42)


# In[27]:


# %%
# Model Building - XGBoost

# Setting parameters


# In[28]:


params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 2

d_train = xgb.DMatrix(X_t, label=y_t)
d_valid = xgb.DMatrix(X_v, label=y_v)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]


# In[29]:


# Training the model


# In[30]:


XGboostModel = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)


# In[31]:


# Saving the model parameters in an external file


# In[32]:


joblib.dump(XGboostModel, path + "XGboostModel.sav")


# In[33]:


# %%
# Model Building - LGBoost

# Setting parameters


# In[34]:


params = {}
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'    
params['eta'] = 0.02
params['max_depth'] = 2

lgtrain = lgb.Dataset(X_t, label=y_t)
lgval = lgb.Dataset(X_v, label=y_v)

LGboostModel = lgb.train(params, lgtrain, 400, valid_sets=[lgtrain, lgval], early_stopping_rounds=50, verbose_eval=10)


# In[35]:


# Saving the model parameters in an external file


# In[36]:


joblib.dump(LGboostModel, path + "LGboostModel.sav")


# In[37]:


# %%
# Model building - CatBoost


# In[38]:


cb_model = CatBoostRegressor(iterations=500, learning_rate=0.02, depth=10, 
                             eval_metric='RMSE', metric_period = 50)
CATboostModel = cb_model.fit(X_t, y_t, eval_set=(X_v, y_v), use_best_model=True, verbose=50)


# In[39]:


# Saving the model parameters in an external file


# In[40]:


joblib.dump(CATboostModel, path + "CATboostModel.sav")


# In[41]:


# %%
# Creating a test data


# In[45]:


test_sub = test.loc[:,["question1", "question2"]].apply(lambda x: x.str.lower())


# In[46]:


test_sub


# In[47]:


x_test = pd.DataFrame()
x_test["X1"] = test_sub.apply(features.word_match_share, axis=1, raw=True)
x_test["X2"] = test_sub.apply(features.tfidf_word_match_share, axis=1, raw=True)
x_test["X3"] = pd.DataFrame(features.common_words_count(test_sub))
X4, X5 = features.fuzz_sort_set(test_sub)
x_test["X4"] = X4
x_test["X5"] = X5


# In[48]:


# %%
# Checking the models on few test data


# In[49]:


XG = joblib.load(path + "XGboostModel.sav")
LG = joblib.load(path + "LGboostModel.sav")
CAT = joblib.load(path + "CATboostModel.sav")


# In[50]:


# XGBoost prediction


# In[51]:


XG_pred = XG.predict(xgb.DMatrix(x_test)) * 100


# In[52]:


# LGBoost prediction


# In[53]:


LG_pred = np.expm1(LG.predict(x_test, num_iteration=LG.best_iteration)) * 100


# In[54]:


# CATBoost prediction


# In[55]:


CAT_pred = np.expm1(CAT.predict(x_test)) * 100


# In[56]:


# %%


# In[58]:


final_pred = test.loc[:, ["question1", "question2", "is_duplicate"]]
final_pred["XGBoost Prediction in Percentage"] = XG_pred
final_pred["LGBoost Prediction in Percentage"] = LG_pred
final_pred["CATBoost Prediction in Percentage"] = CAT_pred
print(final_pred)


# In[ ]:





# In[ ]:




