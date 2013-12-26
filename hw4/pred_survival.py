#!/usr/bin/env python

from sklearn.cross_validation import KFold as KFold
from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np
import pandas as pd

NFOLDS = 10
def train_model(model, X, y, n_folds=NFOLDS):
    """
    A generic KFolds training function,
    returns the best model and the average score
    """
    kf = KFold(n=len(X), n_folds=n_folds, shuffle=True)
    
    scores = []
    best_model = None
    best_score = float("-inf")

    for k, (train_index, test_index) in enumerate(kf):
        X_train, y_train = X.loc[train_index].dropna(), y.loc[train_index].dropna() 
        X_test, y_test = X.loc[test_index].dropna(), y.loc[test_index].dropna()

        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_model = model

    scores_mean = np.mean(scores)
    return (best_model, scores_mean)

def parse(csv, label_col):
    d = pd.read_csv(csv).fillna(0)
    y = d.filter([label_col])
    X = d.drop([label_col],1)
    return X,y

def bulk_predict(model, csv):
    d = pd.read_csv(csv)
    X = d.drop(["PassengerId","Ticket","Name","Cabin","Embarked"],1)
    X = X.fillna(0)
    p = model.predict(X)
    with open("submission.csv","w") as f:
        f.write("PassengerId,Survived\n")
        i = 0
        for pred in p:
            f.write(str(d.ix[i,0])+","+str(pred)+"\n")
            i += 1

if __name__ == "__main__":
    X, y = parse("train.csv","Survived")
    X = X.drop(["PassengerId","Ticket","Name","Cabin","Embarked"],1)
    dtc = DTC(max_depth=3)
    model, score = train_model(dtc,X,y)
    print score
    bulk_predict(model, "test.csv")
