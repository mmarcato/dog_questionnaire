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
from sklearn.feature_selection import SelectFromModel
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


pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('imp', SimpleImputer()),
    ('var', VarianceThreshold()),
    ('uni', FeatureUnion([
        ('skb', SelectKBest()),
        ('linear_pca', PCA()),
        ('kernel_pca', KernelPCA()),
        ('svd', TruncatedSVD()),
        ('fa', FactorAnalysis())
        ])),
    # ('skb', SelectKBest()),
    ('slt', 'passthrough'),
    ('clf', DecisionTreeClassifier(max_features = None, 
                random_state = 0))
    ])
                        
params = {
            'imp__strategy': ['mean', 'median','most_frequent'],

            'uni__skb__score_func' : [f_classif, chi2],
            'uni__skb__k':  [20, 35, 50, 70],  # 114 features in total

            'uni__linear_pca__n_components': [2,5,7],
            'uni__kernel_pca__n_components': [2,5,7],            
            'uni__svd__n_components': [2,5,7],
            'uni__fa__n_components': [2,5,7],

            # 'skb__k':  [20, 35, 50, 'all'],  # min 32 and max 89 features

            'slt' : ['passthrough', 
                    SelectFromModel(LogisticRegression(solver = 'liblinear'))],

            'clf__max_depth': [2, 3, 5]
        }

# cv = LeaveOneOut()
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state = 0)
# cv = StratifiedKFold(n_splits = 3)
# cv = StratifiedShufflesplit(n_splits = 4, test_size = 0.3, random_state = 0)

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
gs_path = os.path.join(dir_model, "prq-dt-1.pkl" )
joblib.dump(gs_results(gs), gs_path, compress = 1 )
print("Model saved to:", gs_path)

