import math
import numpy as np
import pylab as pl
import pandas as pd

from matplotlib.colors import ListedColormap

from sklearn import cross_validation as cv
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error as mse
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.preprocessing import scale


DATA = pd.read_csv("wine.csv")
labels = DATA.filter(['class_id'])
records = DATA.drop(['class_id'],1)

def train_and_classify(nn=5, X = records, y = labels, n_folds=3):
    kf = cv.KFold(n=len(X), n_folds=n_folds, shuffle=True)
    
    accs = []
    for k, (train_idxs, test_idxs) in enumerate(kf):
        # Get all train/test samples for this fold
        print "*"*10 + "kNN" + "*"*10
        print str(train_idxs)
        print str(test_idxs)
        train_X = X.loc[train_idxs]
        train_y = y.loc[train_idxs]
        
        test_X = X.loc[test_idxs]
        test_y = y.loc[test_idxs]

        # Train the model
        model = knn(n_neighbors=nn)
        model.fit(train_X, train_y)

        # Test the model
        acc = model.score(test_X, test_y)
        print str(acc)
        accs.append(acc)

        pred_y = model.predict(test_X)
        cm = confusion_matrix(test_y, pred_y)
        print str(cm)

        # Train the model with LR
        print "*"*10 + "LR" + "*"*10
        modelLR = LR()
        modelLR.fit(train_X, train_y)

        # Test the model with LR
        accLR = modelLR.score(test_X, test_y)
        print str(accLR)

        pred_y = modelLR.predict(test_X)
        cmLR = confusion_matrix(test_y, pred_y)
        print str(cmLR)


def rand_point():
    return np.random.random_sample()*2-1

def rand_sin_sample():
    X = [[rand_point()],[rand_point()]]
    Y = [[math.sin(X[0][0])], [math.sin(X[1][0])]]
    return (X, Y)

def b_bias_variance(n_iterations=1):
    # Minimizing MSE for flat line is just the average height of the two points as the b
    bs = []
    for i in xrange(n_iterations):
        X, Y = rand_sin_sample()
        b = (Y[0][0] + Y[1][0])/2.0
        bs.append(b)
    avg = sum(bs)/len(bs)
    return (avg)

def ax_b_bias_variance(n_iterations=1):
    sas = []
    bs = []
    for i in xrange(n_iterations):
        X, Y = rand_sin_sample()
        a = (Y[0][0] - Y[1][0])/(X[0][0] - X[1][0])
        b = (-1 * a * X[1][0]) + Y[1][0]
        print str(a)
        print str(b)
        sas.append(a)
        bs.append(b)
    avg_a = sum(sas)/len(sas)
    avg_b = sum(bs)/len(bs)
    return (avg_a, avg_b)
    # sample from sin surve
    
