# --------------
import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# path_train : location of test file
# Code starts here
df = pd.read_csv(path_train)
columns=df.columns
def label_race(row):
    for i in range(0,len(columns)):
        if row[i]=='T':
            return columns[i]
df['category'] = df.apply(label_race,axis=1)
df = df[['message','category']]         
print(df.head())


# --------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# Sampling only 1000 samples of each category
df = df.groupby('category').apply(lambda x: x.sample(n=1000, random_state=0))

# Code starts here
all_text = df[['message']]
all_text.message = all_text['message'].str.lower()
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(all_text['message'])
X = tfidf.transform(all_text['message']).toarray()
le = LabelEncoder()
le.fit(df.category)
y = le.transform(df.category)
print(X)
print(y)


# --------------
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Code starts here
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3, random_state=42)
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_val)
log_accuracy = accuracy_score(y_pred,y_val)
print(log_accuracy)

nb = MultinomialNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_val)
nb_accuracy = accuracy_score(y_pred,y_val)
print(nb_accuracy)

lsvm = LinearSVC(random_state=0)
lsvm.fit(X_train,y_train)
y_pred = lsvm.predict(X_val)
lsvm_accuracy = accuracy_score(y_pred,y_val)
print(lsvm_accuracy)


# --------------
# path_test : Location of test data

#Loading the dataframe
df_test = pd.read_csv(path_test)

#Creating the new column category
df_test["category"] = df_test.apply (lambda row: label_race (row),axis=1)

#Dropping the other columns
drop= ["food", "recharge", "support", "reminders", "nearby", "movies", "casual", "other", "travel"]
df_test=  df_test.drop(drop,1)

# Code starts here
all_text = df_test['message'].str.lower()
X_test=tfidf.transform(all_text)
y_test=le.transform(df_test['category'])
y_pred=log_reg.predict(X_test)
log_accuracy_2=accuracy_score(y_pred,y_test)
print(log_accuracy_2)

y_pred=nb.predict(X_test)
nb_accuracy_2=accuracy_score(y_pred,y_test)
print(nb_accuracy_2)

y_pred=lsvm.predict(X_test)
lsvm_accuracy_2=accuracy_score(y_pred,y_test)
print(lsvm_accuracy_2)


# --------------
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim.models.lsimodel import LsiModel
from gensim import corpora
from pprint import pprint
# import nltk
# nltk.download('wordnet')

# Creating a stopwords list
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
# Function to lemmatize and remove the stopwords
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
# Creating a list of documents from the complaints column
list_of_docs = df["message"].tolist()
# Implementing the function for all the complaints of list_of_docs
doc_clean = [clean(doc).split() for doc in list_of_docs]
# Code starts here
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(text) for text in doc_clean]
print(doc_term_matrix)
lsimodel = LsiModel(corpus=doc_term_matrix, num_topics=5, id2word=dictionary)
pprint(lsimodel.print_topics())


# --------------
from gensim.models import LdaModel
from gensim.models import CoherenceModel

# doc_term_matrix - Word matrix created in the last task
# dictionary - Dictionary created in the last task

# Function to calculate coherence values
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    topic_list : No. of topics chosen
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    topic_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(doc_term_matrix, random_state = 0, num_topics=num_topics, id2word = dictionary, iterations=10)
        topic_list.append(num_topics)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return topic_list, coherence_values

# Code starts here
topic_list,coherence_value_list = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=doc_clean, start=1, limit=41, step=5)
print(topic_list)
print(coherence_value_list)
opt_topic=topic_list[coherence_value_list.index(max(coherence_value_list))]
print(opt_topic)
lda_model = LdaModel(corpus=doc_term_matrix, num_topics=opt_topic, id2word = dictionary, iterations=10 , passes=30, random_state=0)
pprint(lda_model.print_topics(5))


