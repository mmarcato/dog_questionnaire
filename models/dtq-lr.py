''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
# ------------------------------------------------------------------------- #
#                           Importing Global Modules                        #
# ------------------------------------------------------------------------- #
import os, sys
import numpy as np
import pandas as pd
from time import time

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FactorAnalysis

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, LeaveOneOut 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------------------- #
#                             Local Imports                                 #    
# ------------------------------------------------------------------------- # 

# Define local directories
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_base = os.path.dirname(dir_current)
sys.path.append(dir_base)

from models.__modules__ import *

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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([ 
        ('ft', DataFrameSelector(feat,'float64')),
        ('var', VarianceThreshold()),
        # ('pca', PCA()),
        ('lr', LogisticRegression(solver = 'liblinear', random_state = 0)) 
    ])

params = {    
        # 'pca__n_components': [1,2,3,4,5],

        'lr__penalty': ['l1', 'l2'],
        'lr__C': np.logspace(-2, 0, 10)
        }

start_time = time()
gs = GridSearchCV(pipe, params, 
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state = 0),
        # cv = LeaveOneOut(), 
        scoring ={"f1": "f1", "roc_auc" : "roc_auc", 
                "accuracy" : "accuracy",
                "precision" : "precision", "recall" :  "recall"},
        refit = 'f1', n_jobs = -1,
        #n_iter = 2500, random_state = 42,
        return_train_score = True, verbose = False )
gs.fit(X,y)
end_time = time()
duration = end_time - start_time
print("--- %s seconds ---" % (duration))
gs_output(gs)


# save gs results to pickle file
gs_path = os.path.join(dir_model, "dtq-lr.pkl" )
joblib.dump(gs_results(gs), gs_path, compress = 1 )
print("Movel saved to:", gs_path)          


gs_loaded = joblib.load(os.path.join(dir_model, "dtq-pca-lr.pkl" ))
gs_output(gs_loaded)