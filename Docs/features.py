import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from fuzzywuzzy import fuzz


#%%
# A function to get the ratio of shared words

def word_match_share(row):
    stops = set(stopwords.words("english"))
    q1words = {}
    q2words = {}    
    for word in str(row[0]).split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row[1]).split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:        
        return 0
    shared_words_in_q1_q2 = [w for w in q1words.keys() if w in q2words]
    
    R = (len(shared_words_in_q1_q2) + len(shared_words_in_q1_q2))/(len(q1words) + len(q2words))
    return R


#%%
# A function to get TF-IDF value


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def tfidf_word_match_share(row):
    stops = set(stopwords.words("english"))
    q1words = {}
    q2words = {}
    for word in str(row[0]).split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row[1]).split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:        
        return 0
        
    train_ques = pd.Series(row[0]).append(pd.Series(row[1])).reset_index(drop=True)
    words = (" ".join(train_ques)).split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    
    if np.sum(total_weights)*1E100 or np.sum(shared_weights)*1E100 > 0.0:
        R = np.sum(shared_weights) / np.sum(total_weights)
    else:
        R = 0    
    
    return R


#%%
# A function which returns number of comman words

def common_words_count(test):
    stop = set(stopwords.words('english'))
    test.question1 = test.question1.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    test.question2 = test.question2.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    test['common_words'] = [set(x[0].split()) & set(x[1].split()) for x in test.loc[:, ['question1', 'question2']].values]
    
    return test.common_words.str.len()



#%%
# A function which returns fuzz_sort_ratio and fuzz_set_ratio

def fuzz_sort_set(test):
    stop = stopwords.words('english')
    test.question1 = test.question1.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    test.question2 = test.question2.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    sort = [fuzz.token_sort_ratio(x[0], x[1]) for x in test.loc[:, ['question1', 'question2']].values]
    sets = [fuzz.token_set_ratio(x[0], x[1]) for x in test.loc[:, ['question1', 'question2']].values]
    
    return sort, sets

