''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
# ------------------------------------------------------------------------- #
#                           Importing Global Modules                        #
# ------------------------------------------------------------------------- #
import os, sys, joblib
from shutil import rmtree
import numpy as np
import pandas as pd
from time import time

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
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
df_inner = pd.read_csv(os.path.join(dir_df, '2022-05-13-df-inner.csv'))
# # selects features
prq_feat = df_inner.columns[2:102].append(df_inner.columns[106:120])
dtq_feat = df_inner.columns[120:151].delete(df_inner.columns[120:151].str.contains("Comments"))
feat = prq_feat.append(dtq_feat)

X = df_inner.loc[:, feat]
y = df_inner.loc[:, 'Outcome'].replace({'Success' : 0, 'Fail': 1})

print("Dataset size:\n{}".format(X.shape))
print("Class imbalance:\n{}".format(df_inner.Outcome.value_counts(normalize=True)))

# ------------------------------------------------------------------------- #
#                             Machine Learning                              #
# ------------------------------------------------------------------------- #

location = "cachedir"
memory = joblib.Memory(location=location, verbose=1)

pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('imp', SimpleImputer(missing_values=np.nan)),
    ('var', VarianceThreshold()),
    ('slt', SelectKBest()),
    ('clf', RandomForestClassifier(n_jobs = 1, 
                max_features = None, 
                random_state = 0))
    ])

params = {
            'slt__score_func' : [f_classif, chi2], 
            'imp__strategy' : ['mean', 'median', 'most_frequent'],
            'slt__k':  [20, 35, 50, 70, 90],  # 140 features in total
            'clf__max_depth': [2, 3, 5],
            'clf__n_estimators': [10, 25, 50, 100]
        }

cv = StratifiedShuffleSplit(n_splits = 4, test_size = 0.3, random_state = 0)

start_time = time()
gs = GridSearchCV(pipe, params, 
        scoring ={"TP": make_scorer(TP), "TN": make_scorer(TN),
                "FP": make_scorer(FP), "FN": make_scorer(FN),
                "f1": "f1", "accuracy" : "accuracy", "roc_auc" : "roc_auc"},
        refit = 'f1', n_jobs = -1, cv = cv, return_train_score = True,
        verbose = False)
gs.fit(X,y)
end_time = time()
duration = end_time - start_time
print("--- %s seconds ---" % (duration))
gs_output(gs)

# save gs results to pickle file
gs_path = os.path.join(dir_model, "inner-2_outcomes.pkl" )
# joblib.dump(gs_results(gs), gs_path, compress = 1 )
print("Model saved to:", gs_path)

# clear cache
memory.clear(warn=False)
rmtree(location)

