# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:00:30 2019

@author: meftahzd
"""
import pandas as pd 
import json
import re
from sklearn.feature_extraction.text import CountVectorizer

with open("publication.json") as f:
    data = json.load(f)["_source"]
    
df=pd.DataFrame(data).T

t=df.loc[:,["abstract","id", "title", "topic"]]

def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

 
t['text'] = t.fillna('').apply(lambda row: row['title'] + row['abstract']+ str(' , '.join(str(v) for v in row["topic"])), axis=1)
t['text'] = t['text'].apply(lambda x:pre_process(x))


def get_stop_words(file_path):
    """load stop words """
    
    with open(file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)
    

#load a set of stop words
stopwords=get_stop_words("stopwords.txt")
 
#get the text column 
docs=t['text'].tolist() 
 
#create a vocabulary of words, 
#ignore words that appear in 85% of documents, 
#eliminate stop words
cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
word_count_vector=cv.fit_transform(docs)