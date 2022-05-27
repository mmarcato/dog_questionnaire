''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
# ------------------------------------------------------------------------- #
#                           Importing Global Modules                        #
# -------------------------------------- ----------------------------------- #
import os, sys
import numpy as np
import pandas as pd
from time import time

# reference https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, confusion_matrix
from sympy import inverse_laplace_transform

import seaborn as sns
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------- #
#                             Local Imports                                 #    
# ------------------------------------------------------------------------- # 

# Define local directories
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_base = os.path.dirname(dir_current)
sys.path.append(dir_current)

%load_ext autoreload
%autoreload 2
from __modules__ import *

# directory where the dataset is located
dir_df = os.path.join(dir_base, 'data')
# directory to save the model
dir_model = os.path.join(dir_base, 'models')


# ------------------------------------------------------------------------- #
#                           Import Dasets from Folder                       #
# ------------------------------------------------------------------------- #

# import all 
dtq = pd.read_csv(os.path.join(dir_df, '2022-05-10-DT_MCPQ-R.csv'))

# selects features
feat = dtq.columns[~dtq.columns.str.contains("Comments")][1:27]
X = dtq.loc[:, feat]
y = dtq.loc[:, 'Outcome'].values

print("Dataset size:\n{}".format(X.shape))
print("Class imbalance:\n{}".format(dtq.Outcome.value_counts(normalize=True)))

# ------------------------------------------------------------------------- #
#                             Machine Learning                              #
# ------------------------------------------------------------------------- #

# old pipeline
pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('var', VarianceThreshold()),
    ('slt', SelectKBest()),
    ('clf', RandomForestRegressor(n_jobs = 1, max_features = None, 
                random_state = 0))
    ])

params = {
            'slt__score_func' : [chi2], # [f_classif, chi2],
            'slt__k':  [7],# [5, 7, 10, 12, 15, 20],  # 26 features in total
            'clf__max_depth': [5], #[2, 3, 5]
            'clf__n_estimators': [50]# [25, 50, 75, 100]
        }

def gs_regression(gs):
    rmse = gs.cv_results_['mean_test_rmse'][gs.best_index_]
    mse = gs.cv_results_['mean_test_mse'][gs.best_index_]
    mae = gs.cv_results_['mean_test_mae'][gs.best_index_]
    print("Best Estimator (rmse) \nTest mean: {:.3f}\t std: {:.3f}\nTrain mean: {:.3f} \t std:  {:.3f}".format(
                -gs.cv_results_['mean_test_rmse'][gs.best_index_],
                gs.cv_results_['std_test_rmse'][gs.best_index_],
                -gs.cv_results_['mean_train_rmse'][gs.best_index_],
                gs.cv_results_['std_train_rmse'][gs.best_index_]))
    print("Best Estimator (mae) \nTest mean: {:.3f}\t std: {:.3f}\nTrain mean: {:.3f} \t std:  {:.3f}\nparameters: {}".format(
                -gs.cv_results_['mean_test_mae'][gs.best_index_],
                gs.cv_results_['std_test_mae'][gs.best_index_],
                -gs.cv_results_['mean_train_mae'][gs.best_index_],
                gs.cv_results_['std_train_mae'][gs.best_index_],
                gs.best_params_))

import random
def save(y_true, y_pred):
    true = y_true.ravel()
    pred = y_pred.ravel()
    df = pd.DataFrame({'true' : true, 'pred' : pred})
    df.to_csv("..//results//cross_val.csv", mode='a', index=False)
    return(random.uniform(0, 1))

folds = 3
cv = StratifiedKFold(n_splits = folds)

start_time = time()
gs = GridSearchCV(pipe, param_grid = params, 
        scoring ={
                "rmse": "neg_root_mean_squared_error", 
                "save": make_scorer(save),
                "mse": "neg_mean_squared_error",
                "mae":"neg_mean_absolute_error"},
        refit = 'mae', n_jobs = 1, cv = cv, return_train_score = True)
gs.fit(X,y)
end_time = time()
duration = end_time - start_time
print("--- %s seconds ---" % (duration))

# print gs results
gs_regression(gs)


# y_pred = gs.best_estimator_.predict(X)
# save(y.ravel(), y_pred)
df_cross = pd.read_csv("..//results//cross_val.csv")
print("Cross-validation results shape: {}".format(df_cross.shape))

# create a columns with the set name
idx = df_cross.index[df_cross.true == 'true'].insert(0, 0)
set = ['Test', 'Train'] * int((len(idx) + 1)/2)
df_cross['set'] = pd.Series(df_cross.index.map(pd.Series(set, index = idx)), 
                        index = df_cross.index).fillna(method = "ffill")

# drop row with the header
df_cross = df_cross[df_cross.true != 'true']
print("Cross-validation results shape: {}".format(df_cross.shape))

# converting data in true and pred
print(df_cross.dtypes)
df_cross.loc[:,['true','pred']] = df_cross[['true','pred']].astype('float64')
print(df_cross.dtypes)


# boxplot of test set prediction vs true label
ax = sns.stripplot(x='true', y='pred', hue = 'set', dodge = True, data=df_cross, color = "0.25")
ax = sns.boxplot(x='true', y='pred', hue = 'set', data=df_cross)


df_cross['label'] = df_cross.pred.round()
print("Predicted class in the test set:\n{}".format(df_cross.label[df_cross.set == 'Test'].value_counts()))

def metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, \
                labels  = np.sort(np.unique(y_true)))
    
    fp = cm.sum(axis=0) - np.diag(cm) 
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    tp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)

    ACC = (tp+tn)/(tp+tn+fp+fn)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = tp/(tp+fn)
    # Specificity or true negative rate
    TNR = tn/(tn+fp) 
    # Precision or positive predictive value
    PPV = tp/(tp+fp)
    
    df_metrics = pd.DataFrame({'Labels': np.sort(np.unique(y_true)),
                    'ACC': ACC, 'TPR': TPR, 'TNR': TNR, 'PPV' : PPV,
                    'F1' : f1_score(y_true, y_pred, labels = np.sort(np.unique(y_true)), average = None),
                    'Precision': precision_score(y_true, y_pred, labels = np.sort(np.unique(y_true)), average = None)
                    })
    print(df_metrics)
        
    # print("\nACC: {:.2f}, TPR: {:.2f}, TNR: {:.2f}, PPV: {:.2f}".format(ACC, TPR, TNR, PPV))
    print("ACC: {:.2f}".format(accuracy_score(y_true, y_pred)))
                # precision_score(df_cross.true, df_cross.label))
    print("f1_macro: {:.2f}, f1_micro: {:.2f}, f1_weighted: {:.2f}".format(
            f1_score(y_true, y_pred, average = 'macro'),
            f1_score(y_true, y_pred, average = 'micro'),
            f1_score(y_true, y_pred, average = 'weighted')))

metrics(df_cross.true, df_cross.label)

from sklearn.metrics import accuracy_score, f1_score, precision_score

