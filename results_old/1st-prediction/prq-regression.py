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
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, confusion_matrix
from sympy import inverse_laplace_transform

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
prq = pd.read_csv(os.path.join(dir_df, '2022-05-10-PR_C-BARQ.csv'))
# selects features
feat = prq.columns[2:102].append(prq.columns[106:120])

X = prq.loc[:, feat]
y = prq.loc[:,[ 'Label']]

print("Dataset size:\n{}".format(X.shape))
print("Class imbalance:\n{}".format(y.value_counts(normalize=True)))
# my dataset is too small to have a hold out set
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

fig, ax = plt.subplots(figsize=(20,10))         
corr = X[feat[:-14]].corr()
sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=14)
plt.show()


fig, ax = plt.subplots(figsize=(20,10))         
corr = X[feat[-14:]].corr()
sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=14)
plt.show()

# ------------------------------------------------------------------------- #
#                             Machine Learning                              #
# ------------------------------------------------------------------------- #

# old pipeline
pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('imp', SimpleImputer(missing_values=np.nan)),
    ('var', VarianceThreshold()),
    ('slt', SelectKBest()),
    ('clf', RandomForestRegressor(n_jobs = 1, max_features = None, 
                random_state = 0))
    ])

params = {
            # 'slt__score_func' : [chi2], 
            # 'imp__strategy' : ['mean'],
            # 'slt__k':  [7],
            # 'clf__max_depth': [5],
            # 'clf__n_estimators': [50],
            'slt__score_func' : [f_classif, chi2],
            'imp__strategy' : ['mean', 'median', 'most_frequent'],
            'slt__k':  [5, 7, 10, 12, 15, 20], 
            'clf__max_depth': [2, 3, 5],
            'clf__n_estimators': [25, 50, 75, 100]
        }

folds = 4
cv = StratifiedShuffleSplit(n_splits = folds, test_size = 0.3, random_state = 0)

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
# print gs results
gs_regression(gs)

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


metrics(df_cross.true, df_cross.label)

from sklearn.metrics import accuracy_score, f1_score, precision_score

