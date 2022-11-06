#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('C:\\Users\\Peace\\Documents\\Project\\Final\\')
import features


# In[17]:


path = "C:\\Users\\Peace\\Documents\\Project\\Final\\"


# In[18]:


# %%

print("Welcome user! \n")


# In[19]:


def ask():
    ques = pd.DataFrame({'question1':[''],
                        'question2':['']})    
    ques.loc[0, 'question1'] = input("Enter your first question: ")
    ques.loc[0, 'question2'] = input("Enter your second question: ")
    
    return ques


# In[20]:


def model():    
    print("Choose a model to get the result")
    print("1. XG Boost")
    print("2. LG Boost")
    print("3. CAT Boost (Best model as per test results)")
    
    choice = int(input())
    
    if choice != 1 and choice != 2 and choice != 3:
        print("Wrong model choice!")
        return -1
    return choice


# In[21]:


def make_features(ques):
    x_test = pd.DataFrame()
    x_test["X1"] = ques.apply(features.word_match_share, axis=1, raw=True)
    x_test["X2"] = ques.apply(features.tfidf_word_match_share, axis=1, raw=True)
    x_test["X3"] = pd.DataFrame(features.common_words_count(ques))
    X4, X5 = features.fuzz_sort_set(ques)
    x_test["X4"] = X4
    x_test["X5"] = X5
    return x_test


# In[22]:


out = 0
while(1):
    ques = ask()
    x_test = make_features(ques)    
    
    while(1):
        model_choice = model()        
        if model_choice != -1:
                if model_choice == 1:
                    XG = joblib.load(path + "XGboostModel.sav")
                    print("Both the questions have ", np.round(XG.predict(xgb.DMatrix(x_test)) * 100,2)[0],
                          "% similar intent")
                    break
                elif model_choice == 2:
                    LG = joblib.load(path + "LGboostModel.sav")
                    print("Both the questions have ", 
                          np.round(np.expm1(LG.predict(x_test, num_iteration=LG.best_iteration)) * 100,2)[0],
                          "% similar intent")
                    break
                elif model_choice == 3:
                    CAT = joblib.load(path + "CATboostModel.sav")
                    print("Both the questions have ", 
                          np.round(np.expm1(CAT.predict(x_test)) * 100,2)[0],
                          "% similar intent")
                    break
    val = input("Do you want to try again with new question set? y/n: ")
    if val == 'n':
        break


# In[ ]:





# In[ ]:




