import json
import pandas as pd 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("/data/json/publication.json") as f :
    d = json.load(f)
    
s=pd.DataFrame(d).T

t=s.loc[:,["abstract","id", "title", "topic"]]


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

count_vectorizer = TfidfVectorizer(stop_words='english')
count_vectorizer = TfidfVectorizer()

sparse_matrix = count_vectorizer.fit_transform(t['text'].values.tolist())
doc_term_matrix = sparse_matrix.todense()

df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names())

output = pd.DataFrame(data=cosine_similarity(df, df), index= t.index, columns=t.index)

output.to_pickle('my_file.pkl')





