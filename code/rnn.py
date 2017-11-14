import matplotlib.pyplot as plt
import re
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# https://radimrehurek.com/data_science_python/

def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

# Create dataset that we interface using pandas
dataset = []
with open('../data/label-abstract.txt') as f:
    data = f.read().split('\n\n')
    for example in data:
        components = re.split("\<[a-z]+\>", example)
        if len(components) > 1:
            idnum = components[1].strip()
            category = components[2].strip()
            abstract = components[3].strip().replace('\n', ' ')
            dataset.append((idnum, category, abstract))
df = pandas.DataFrame(data = dataset, columns= ['ID', 'Category', 'Abstract'])

# Represent each abstract as a list of tokens (lemmas) then covert into vector
# In a bag-of-words model:
# 1. Count how many times a word occurs in each abstract (term frequency)
# 2. Weight the counts so that frequent tokens across abstracts get lower weight (inverse document frequency)
# 3. Normalize the vectors to unit length to abstract from original text length (L2 norm)
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(df.Abstract)
abstracts_bow  = bow_transformer.transform(df.Abstract)
tfidf_transformer = TfidfTransformer().fit(abstracts_bow)
abstracts_tfidf = tfidf_transformer.transform(abstracts_bow)
classifier = MLPClassifier().fit(abstracts_tfidf, df.Category)
all_predictions = classifier.predict(abstracts_tfidf)

print 'accuracy', accuracy_score(df.Category, all_predictions)
print classification_report(df.Category, all_predictions)
