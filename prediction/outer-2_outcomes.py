''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
# ------------------------------------------------------------------------- #
#                           Importing Global Modules                        #
# ------------------------------------------------------------------------- #
import os, sys
from tkinter import Grid
import numpy as np
import pandas as pd
from time import time

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer

# ------------------------------------------------------------------------- #
#                             Local Imports                                 #    
# ------------------------------------------------------------------------- # 

# Define local directories
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_base = os.path.dirname(dir_current)
sys.path.append(dir_current)

from __modules__ import *

# directory where the dataset is located
dir_df = os.path.join(dir_base, 'data')
# directory to save the model
dir_model = os.path.join(dir_base, 'models')


# ------------------------------------------------------------------------- #
#                           Import Dasets from Folder                       #
# ------------------------------------------------------------------------- #

# import all 
df_outer = pd.read_csv(os.path.join(dir_df, '2022-05-13-df-outer.csv'))
# # selects features
prq_feat = df_outer.columns[2:102].append(df_outer.columns[106:120])
dtq_feat = df_outer.columns[120:151].delete(df_outer.columns[120:151].str.contains("Comments"))
feat = prq_feat.append(dtq_feat)

X = df_outer.loc[:, feat]
y = df_outer.loc[:, 'Outcome'].values

print("Dataset size:\n{}".format(X.shape))
print("Class imbalance:\n{}".format(df_outer.Outcome.value_counts(normalize=True)))
# my dataset is too small to have a hold out set
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# ------------------------------------------------------------------------- #
#                             Machine Learning                              #
# ------------------------------------------------------------------------- #

pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('imp', 'passthrough'),
    ('var', VarianceThreshold()),
    ('slt', SelectKBest()),
    ('clf', RandomForestClassifier(n_jobs = 1, max_features = None, 
                random_state = 0))
    ])

# gs_simple_imp = GridSearchCV(SimpleImputer(), 
#                         {'strategy' : ['mean', 'median', 'uniform']})

# gs_KNN_imp = GridSearchCV(KNNImputer(), 
#                         {'weights' : ['uniform', 'distance']})
params = {
            'slt__score_func' : [f_classif, chi2], 
            'imp' : [SimpleImputer(strategy = 'mean'), 
                        SimpleImputer(strategy = 'median'),
                        SimpleImputer(strategy = 'most_frequent'),
                        KNNImputer(weights = 'uniform'),
                        KNNImputer(weights = 'distance'),
                        ],
        #     'imp__strategy' : ['mean', 'median', 'most_frequent'],
            'slt__k':  [20, 35, 50, 70, 90, 115],  # 140 features in total
            'clf__max_depth': [2, 3, 5],
            'clf__n_estimators': [10, 25, 50, 100]
        }

cv = StratifiedKFold(n_splits = 3)

start_time = time()
gs = GridSearchCV(pipe, param_grid = params, 
        scoring ={"TP": make_scorer(TP), "TN": make_scorer(TN),
                "FP": make_scorer(FP), "FN": make_scorer(FN),
                "f1_macro": "f1_macro", "f1_weighted": "f1_weighted",
                "f1_micro": "f1_micro", "accuracy" : "accuracy"},
        refit = 'accuracy', n_jobs = -1, cv = cv, return_train_score = True)
gs.fit(X,y)
end_time = time()
duration = end_time - start_time
print("--- %s seconds ---" % (duration))

# print gs results
gs_output(gs)