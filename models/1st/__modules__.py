''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------------------------------------------- #
#                                   Classes                                 #
# ------------------------------------------------------------------------- # 

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, dtype=None):
        self.attribute_names = attribute_names
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_selected = X.loc[:,self.attribute_names]
        if self.dtype:
            return X_selected.astype(self.dtype).values
        return X_selected.values

# ------------------------------------------------------------------------- #
#                                  Evaluation                               #
# ------------------------------------------------------------------------- # 

class gs_results:
    # Storing Grid Search results
    def __init__(self, gs):
        self.cv_results_ = gs.cv_results_
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_
        self.best_index_ = gs.best_index_

# THIS CODE HAS BEEN CHANGED AND NEEDS TO BE RE-EVALUATED IF USED IN THE FUTURE

def TN(y_true, y_pred):
    # Calculate True Negative from confusion matrix
    cm = confusion_matrix(y_true, y_pred, \
                labels  = np.sort(np.unique(y_true)))
    return(cm[0,0])

def FP(y_true, y_pred):
    # Calculate False Positive from confusion matrix
    cm = confusion_matrix(y_true, y_pred, \
                labels  = np.sort(np.unique(y_true)))
    return(cm[0,1])

def FN(y_true, y_pred):
    # Calculate False Negative from confusion matrix
    cm = confusion_matrix(y_true, y_pred, \
                labels  = np.sort(np.unique(y_true)))
    return(cm[1,0])

def TP(y_true, y_pred):
    # Calculate True Positive from confusion matrix
    cm = confusion_matrix(y_true, y_pred, \
                labels  = np.sort(np.unique(y_true)))
    return(cm[1,1])


# FOR THE MULTI-CLASS PROBLEM
def gs_output(gs):
    print("Best Estimator (accuracy) \nTest mean: {:.3f} std: {:.3f}\nTrain mean: {:.3f} std: {:.3f}\n\nparameters: {}\n".format(
                gs.cv_results_['mean_test_accuracy'][gs.best_index_],
                gs.cv_results_['std_test_accuracy'][gs.best_index_],
                gs.cv_results_['mean_train_accuracy'][gs.best_index_],
                gs.cv_results_['std_train_accuracy'][gs.best_index_],
                gs.best_params_))
                
    print("Best Estimator (f1-score) \nTest mean: {:.3f} std: {:.3f}\nTrain mean: {:.3f} std: {:.3f}".format(
                gs.cv_results_['mean_test_f1'][gs.best_index_],
                gs.cv_results_['std_test_f1'][gs.best_index_],
                gs.cv_results_['mean_train_f1'][gs.best_index_],
                gs.cv_results_['std_train_f1'][gs.best_index_]))
    
    tp = gs.cv_results_['mean_test_TP'][gs.best_index_]
    tn = gs.cv_results_['mean_test_TN'][gs.best_index_]
    fp = gs.cv_results_['mean_test_FP'][gs.best_index_]
    fn = gs.cv_results_['mean_test_FN'][gs.best_index_]
    
    ACC = (tp+tn)/(tp+tn+fp+fn)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = tp/(tp+fn)
    # Specificity or true negative rate
    TNR = tn/(tn+fp) 
    # Precision or positive predictive value
    PPV = tp/(tp+fp)
    
    print("\nACC: {:.2f}, ROC_AUC: {:.2f},\nTPR: {:.2f}, TNR: {:.2f}\nPPV: {:.2f}".format(
            ACC, gs.cv_results_['mean_test_roc_auc'][gs.best_index_],TPR, TNR, PPV))
        
    print("f1_macro: {:.2f}, f1_micro: {:.2f}, f1_weighted: {:.2f}".format(
            gs.cv_results_['mean_test_f1_macro'][gs.best_index_],
                gs.cv_results_['mean_test_f1_micro'][gs.best_index_],
                gs.cv_results_['mean_test_f1_weighted'][gs.best_index_]))

# FOR THE REGRESSION PROBLEM

import random
def save(y_true, y_pred):
    true = y_true.ravel()
    pred = y_pred.ravel()
    df = pd.DataFrame({'true' : true, 'pred' : pred})
    df.to_csv("..//results//cross_val.csv", mode='a', index=False)
    return(random.uniform(0, 1))


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
            