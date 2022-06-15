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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

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
dtq = pd.read_csv(os.path.join(dir_df, '2022-05-10-DTQ_MCPQ-R.csv'))

# selects features
feat = dtq.columns[~dtq.columns.str.contains("Comments")][1:27]
X = dtq.loc[:, feat]
y = dtq.loc[:, 'Outcome'].replace({'Success' : 0, 'Fail': 1})

print("Dataset size:\n{}".format(X.shape))
print("Class imbalance:\n{}".format(dtq.Outcome.value_counts(normalize=True)))

# ------------------------------------------------------------------------- #
#                             Machine Learning                              #
# ------------------------------------------------------------------------- #

location = "cachedir"
memory = joblib.Memory(location=location, verbose=1)

pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('var', VarianceThreshold()),
    ('slt', SelectKBest()),
    ('clf', RandomForestClassifier(n_jobs = 1, max_features = None, 
                random_state = 0))
    ], memory = memory)

params = {
            'slt__score_func' : [f_classif, chi2], 
            'slt__k':  [5, 7, 10, 12, 15],  # 26 features in total
            'clf__max_depth': [2, 3, 5],
            'clf__n_estimators': [25, 50, 75, 100]
        }

cv = StratifiedKFold(n_splits = 3)

start_time = time()
gs = GridSearchCV(pipe, param_grid = params, 
        scoring ={"f1": "f1", "roc_auc" : "roc_auc", "accuracy" : "accuracy",
                "precision" : "precision", "recall" :  "recall"},
        refit = 'f1', n_jobs = -1, cv = cv, 
        return_train_score = True, verbose = False )
gs.fit(X,y)
end_time = time()
duration = end_time - start_time
print("--- %s seconds ---" % (duration))        
gs_output(gs)


# save gs results to pickle file
gs_path = os.path.join(dir_model, "dt-2_outcomes-1.pkl" )
joblib.dump(gs_results(gs), gs_path, compress = 1 )
print("Movel saved to:", gs_path)

# clear cache
memory.clear(warn=False)
rmtree(location)