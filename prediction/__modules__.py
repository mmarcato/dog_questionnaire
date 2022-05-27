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

def FP(y_true, y_pred):
    # Calculate False Positive from confusion matrix
    cm = confusion_matrix(y_true, y_pred, \
                labels  = np.sort(np.unique(y_true)))
    return(cm[1,0])

def FN(y_true, y_pred):
    # Calculate False Negative from confusion matrix
    cm = confusion_matrix(y_true, y_pred, \
                labels  = np.sort(np.unique(y_true)))
    return(cm[0,1])

def TN(y_true, y_pred):
    # Calculate True Negative from confusion matrix
    cm = confusion_matrix(y_true, y_pred, \
                labels  = np.sort(np.unique(y_true)))
    return(cm[0,0])

def TP(y_true, y_pred):
    # Calculate True Positive from confusion matrix
    cm = confusion_matrix(y_true, y_pred, \
                labels  = np.sort(np.unique(y_true)))
    return(cm[1,1])


def gs_output(gs):
    print("Best Estimator (f1) \nTest mean: {:.3f} std: {:.3f}\nTrain mean: {:.3f} std:  {:.3f}\n\nparameters: {}".format(
                gs.cv_results_['mean_test_f1'][gs.best_index_],
                gs.cv_results_['std_test_f1'][gs.best_index_],
                gs.cv_results_['mean_train_f1'][gs.best_index_],
                gs.cv_results_['std_train_f1'][gs.best_index_],
                gs.best_params_))
    
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


    # OLD PRINTS
    print("Best Estimator (accuracy) \nTest mean: {:.3f}\t std: {:.3f}\nTrain mean: {:.3f} \t std:  {:.3f}\nparameters: {}".format(
                gs.cv_results_['mean_test_accuracy'][gs.best_index_],
                gs.cv_results_['std_test_accuracy'][gs.best_index_],
                gs.cv_results_['mean_train_accuracy'][gs.best_index_],
                gs.cv_results_['std_train_accuracy'][gs.best_index_],
                gs.best_params_))
                
        
    # print("f1_macro: {:.2f}, f1_micro: {:.2f}, f1_weighted: {:.2f}".format(
    #         gs.cv_results_['mean_test_f1_macro'][gs.best_index_],
    #             gs.cv_results_['mean_test_f1_micro'][gs.best_index_],
    #             gs.cv_results_['mean_test_f1_weighted'][gs.best_index_]))
