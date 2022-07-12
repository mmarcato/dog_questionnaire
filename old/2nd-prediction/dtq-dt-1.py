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

pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
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
# hyperparameter search space = 2**3 * 3**5 * 4
params = {
                'uni__skb__score_func' : [f_classif, chi2],
                'uni__skb__k':  [5, 7, 10, 15],  # 26 features in total

                'uni__linear_pca__n_components': [2,5,7],
                'uni__kernel_pca__n_components': [2,5,7],            
                'uni__svd__n_components': [2,5,7],
                'uni__fa__n_components': [2,5,7],

                # 'skb__k':  [10, 13, 'all'],  # min 17 and max 43 features
                'slt' : ['passthrough', 
                        SelectFromModel(LogisticRegression(solver = 'liblinear'))],

                'clf__max_depth': [2, 3, 5]
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
gs_path = os.path.join(dir_model, "dtq-dt-1.pkl" )
joblib.dump(gs_results(gs), gs_path, compress = 1 )
print("Movel saved to:", gs_path)

