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
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FactorAnalysis
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

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
prq = pd.read_csv(os.path.join(dir_df, '2022-05-10-PRQ_C-BARQ.csv'))

# selects features
feat = prq.columns[2:102].append(prq.columns[106:120])
X = prq.loc[:, feat]
y = prq.loc[:, 'Outcome'].replace({'Success' : 0, 'Fail': 1})

print("Dataset size:\n{}".format(X.shape))
print("Class imbalance:\n{}".format(prq.Outcome.value_counts(normalize=True)))

# ------------------------------------------------------------------------- #
#                             Machine Learning                              #
# ------------------------------------------------------------------------- #

from sklearn.preprocessing import StandardScaler

pipe = Pipeline([ 
        ('ft', DataFrameSelector(feat,'float64')),
        # ('slc', StandardScaler()),
        ('imp', SimpleImputer()),
        # ('imp', KNNImputer()),
        ('var', VarianceThreshold()),
        ('pca', PCA()),
        ('lr', LogisticRegression(solver = 'liblinear', random_state = 0)) 
    ])

                        
params = {
        'imp__strategy': ['mean', 'median','most_frequent'],

        # 'imp__weights' : ['uniform', 'distance'],
        # 'imp__n_neighbors' : [2,3,4,5],

        'pca__n_components': [1,2,3,4,5],

        'lr__penalty': ['l1', 'l2'],
        'lr__C': np.logspace(-2, 0, 10)
        }

start_time = time()
gs = GridSearchCV(pipe, params, 
        cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=5, random_state = 0),
        scoring ={"f1": "f1", "roc_auc" : "roc_auc", 
                "accuracy" : "accuracy",
                "precision" : "precision", "recall" :  "recall"},
        refit = 'f1', n_jobs = -1, 
        # n_iter = 10, random_state = 42,
        return_train_score = True, verbose = False )
gs.fit(X,y)
end_time = time()
duration = end_time - start_time
print("--- %s seconds ---" % (duration))        
gs_output(gs)

# save gs results to pickle file
gs_path = os.path.join(dir_model, "prq-spl-pca-lr.pkl" )
joblib.dump(gs_results(gs), gs_path, compress = 1 )
print("Model saved to:", gs_path)

