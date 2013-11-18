#!/usr/bin/env python

import numpy as np
import pandas as pd
import pylab as pl      # note this is part of matplotlib

from sklearn import cross_validation as cv
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_curve, auc

DATASET = "logit-train.csv"
NUM_FOLDS = 10
DATA = pd.read_csv(DATASET).dropna()
LABEL_CATEGORY = 'heartdisease::category|0|1'
LABELS = DATA.filter([LABEL_CATEGORY])
FEATURES = DATA.drop([LABEL_CATEGORY],1)

def avg_auc_by_feature():
    total_auc = 0
    auc_count = 0
    for auc,_ in train_and_classify_by_features():
        total_auc += auc
        auc_count += 1
    return float(total_auc)/auc_count

def train_and_classify_by_features(data=FEATURES, n_folds=5, labels=LABELS):
    kf = cv.KFold(n=len(data.columns), n_folds=n_folds, shuffle=True)

    # Use nested cross-validation to try out feature combinations
    for k, (feature1_set_idx, feature2_set_idx) in enumerate(kf):

        # Select for ~11 features at a time
        col_select = feature1_set_idx
        data_filtered = data.iloc[:,feature1_set_idx]
        
        # Yield the avg_auc for the dataset
        avg_auc = train_and_classify(features=data_filtered, labels=labels)[0]
        features = data_filtered.columns
        yield (avg_auc, features)
        

def train_and_classify(n_folds=NUM_FOLDS, features=FEATURES, labels=LABELS):
    kf = cv.KFold(n=len(features), n_folds=n_folds, shuffle=True)

    num_misclassified = 0
    num_classified = 0
    num_iterations = 0
    auc_total = 0

    for k, (train_set_idx, test_set_idx) in enumerate(kf):
        train_features = features.loc[train_set_idx].dropna()
        train_labels = labels.loc[train_set_idx].dropna()

        test_features = features.loc[test_set_idx].dropna()        
        test_labels = labels.loc[test_set_idx].dropna()

        model = LR()
        model.fit(train_features, train_labels)
        
        pred_labels = model.predict(test_features)

        num_classified += len(pred_labels)
        num_misclassified += [x==y for x,y in zip(pred_labels, test_labels.values.flatten())].count(False)

        # ROC/AUC
        fpr, tpr, thresholds = roc_curve(test_labels, pred_labels, pos_label=1)
        
        roc_auc = auc(fpr, tpr)
        auc_total += roc_auc
        num_iterations += 1

        # Uncomment this to use as an iterator
        #plot_roc(fpr, tpr, roc_auc)
        #yield 

    avg_auc = float(auc_total)/num_iterations
    pct_misclassified = float(num_misclassified)/num_classified

    #yield (avg_auc, pct_misclassified)
    return (avg_auc, pct_misclassified)

def plot_roc(fpr, tpr, roc_auc):
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()    
