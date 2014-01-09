#!/usr/bin/env python

import pandas as pd
import numpy as np
from dateutil import parser
from nltk import NaiveBayesClassifier as nbc
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn import cross_validation as cv
import nltk
from collections import Counter

N_FOLDS = 10
N_GRAM = 1

def extract_features(message):
    words = nltk.word_tokenize(message)
    # Note: simplification for now, ignore anything with any punctuation (though things like "don't" are wiped out for now)
    words = [word.lower() for word in words if word.isalnum()]
    return {word: True for word in words}

def extract_ngram_features(message, n=N_GRAM):
    words = nltk.word_tokenize(message)
    words = [word.lower() for word in words if word.isalnum()]
    ws = set()
    for i in xrange(n):
        ws = ws.union(set([word for word in nltk.ingrams(words, i+1)]))
    return {word: True for word in ws}
    
def bulk_extract_features(messages):
    return map(lambda x: extract_ngram_features(x,N_GRAM), messages)

def train_nltk(data, labels):
    '''
    Returns a trained nltk.NaiveBayesClassifier
    
    Inputs
    ---------
    data -- np.array of tuples
    '''
    # For now, shuffle, since for now assuming that only the post language itself is all that's needed for offensive measure, though in the future, 2 anti-something users may actually not be offended by one another if they are both negative about something
    kf = cv.KFold(n=len(data), n_folds=N_FOLDS, shuffle=True)

    best_model = None
    max_acc = float('-inf')
    for k, (train_index, test_index) in enumerate(kf):
        X_train, Y_train = data[train_index], labels[train_index]
        X_test, Y_test = data[test_index], labels[test_index]

        features_train = bulk_extract_features(X_train)
        features_test = bulk_extract_features(X_test)

        train_set = zip(features_train, Y_train)
        test_set = zip(features_test, Y_test)
        
        model = nbc.train(train_set)

        acc = nltk.classify.accuracy(model, test_set)
        print str(acc)
        if acc > max_acc:
            max_acc = acc
            best_model = model
    best_model.show_most_informative_features(30)
    return best_model

if __name__ == "__main__":
    data = pd.read_csv("nb_train.csv",sep=",")
    labels = data.filter(["Insult"])
    records = data.drop(["Insult"],1)

    nbc_model = train_nltk(records, labels)
    pred_f = pd.read_csv("nb_predict.csv",sep=",")

    map(nbc_model.classify, pred_f)
