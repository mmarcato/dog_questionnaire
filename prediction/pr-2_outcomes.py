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
prq = pd.read_csv(os.path.join(dir_df, '2022-05-10-PR_C-BARQ.csv'))
# # selects features
feat = prq.columns[2:102].append(prq.columns[106:120])

X = prq.loc[:, feat]
y = prq.loc[:, 'Outcome'].values

print("Dataset size:\n{}".format(X.shape))
print("Class imbalance:\n{}".format(prq.Outcome.value_counts(normalize=True)))
# my dataset is too small to have a hold out set
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# ------------------------------------------------------------------------- #
#                             Machine Learning                              #
# ------------------------------------------------------------------------- #

pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('imp', SimpleImputer(missing_values=np.nan)),
    ('var', VarianceThreshold()),
    ('slt', SelectKBest()),
    ('clf', RandomForestClassifier(n_jobs = 1, max_features = None, 
                random_state = 0))
    ])

params = {
            'slt__score_func' : [f_classif, chi2], 
            'imp__strategy' : ['mean', 'median', 'most_frequent'],
            'slt__k':  [20, 35, 50, 70, 90],  # 114 features in total
            'clf__max_depth': [2, 3, 5],
            'clf__n_estimators': [10, 25, 50, 100]
        }

# cv = StratifiedKFold(n_splits = 2)
cv = StratifiedShuffleSplit(n_splits = 4, test_size = 0.3, random_state = 0)

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

# # save gs results to pickle file
# run = "model1"
# gs_path = os.path.join(dir_model, run)
# print(gs_path)
# joblib.dump(gs_results(gs), gs_path, compress = 1 )


# ------------------------------------------------------------------------- #
#                            Feature Importance                             #
# ------------------------------------------------------------------------- #
# load gs results from pickle file

# feature importances dataframe
ft = pd.DataFrame({'Name' : feat})

# Above Variance Threshold
ft['Var_Flag'] = False
ft.loc[gs.best_estimator_['var'].variances_ > 0, 'Var_Flag'] = True

# Select K Best - Scores
ft['SKB_Scores'] = np.nan
ft.loc[gs.best_estimator_['var'].variances_ > 0,'SKB_Scores'] = gs.best_estimator_['slt'].scores_

# creating a flag for features that were selected by skb
ft['SKB_Flag'] = False
ft.loc[ft.nlargest(gs.best_params_['slt__k'],['SKB_Scores']).index, 'SKB_Flag'] = True

# Random Forest
ft['RF_Importances'] = np.nan
ft.loc[ft.SKB_Flag == 1, 'RF_Importances'] = gs.best_estimator_['clf'].feature_importances_


# plot feature importance
from matplotlib import pyplot as plt
plt.figure(figsize=(10,5))
width = 0.3
idx = np.arange(ft.SKB_Flag.sum())

# feature scores for those selected by SKB
skb_norm = ft.loc[ft.SKB_Flag == True, "SKB_Scores"] / ft.loc[ft.SKB_Flag == True, "SKB_Scores"].sum()
plt.bar(idx, skb_norm, width, label = "SKB Scores")
# feature importances for those selected by SKB
plt.bar(idx + width, ft.loc[ft.SKB_Flag == True, "RF_Importances"], width, label='RF Importances')
plt.xticks(rotation = 45)

plt.xlabel('Features - items and factors numbers')
plt.ylabel('Normalised values')
plt.title('SKB Scores and RF Importances for the {} highest scoring features'.format(ft.SKB_Flag.sum()))
plt.xticks(idx + width / 2, ft.loc[ft.SKB_Flag == True, "Name"].str.split().str[0])

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()

#### DUMMY CLASSIFIER

from sklearn.dummy import DummyClassifier
pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('imp', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('var', VarianceThreshold()),
    ('slt', SelectKBest(score_func = f_classif)),
    ('clf', DummyClassifier("most_frequent", random_state = 0))
    ])

params = {
            'slt__k':  [10, 20, 35, 55, 80],  
}
