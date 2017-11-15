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
from sklearn.model_selection import train_test_split

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
# 2. Weigh counts so that frequent tokens across abstracts get lower weight (inverse document frequency)
# 3. Normalize vectors to abstract from original text length (L2 norm)
trainData, testData = train_test_split(df, test_size=0.1)

vectorizer = CountVectorizer(analyzer=split_into_lemmas)
tfidfTransformer = TfidfTransformer()
classifier = MLPClassifier()

abstracts_bow_train = vectorizer.fit_transform(trainData.Abstract)
abstracts_tfidf_train = tfidfTransformer.fit_transform(abstracts_bow_train)
classifier = classifier.fit(abstracts_tfidf_train, trainData.Category)

abstracts_bow_test = vectorizer.transform(testData.Abstract)
abstracts_tfidf_test = tfidfTransformer.transform(abstracts_bow_test)
test_predictions = classifier.predict(abstracts_tfidf_test)

print 'accuracy', accuracy_score(testData.Category, test_predictions)
print classification_report(testData.Category, test_predictions)
