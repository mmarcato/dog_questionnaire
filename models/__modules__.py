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

def gs_output(gs):
    # OLD PRINTS
    print("Best Estimator (accuracy) \nTest mean: {:.3f} std: {:.3f}\nTrain mean: {:.3f} std: {:.3f}\n".format(
                gs.cv_results_['mean_test_accuracy'][gs.best_index_],
                gs.cv_results_['std_test_accuracy'][gs.best_index_],
                gs.cv_results_['mean_train_accuracy'][gs.best_index_],
                gs.cv_results_['std_train_accuracy'][gs.best_index_]))
                
    print("Best Estimator (f1-score) \nTest mean: {:.3f} std: {:.3f}\nTrain mean: {:.3f} std: {:.3f}\n".format(
                gs.cv_results_['mean_test_f1'][gs.best_index_],
                gs.cv_results_['std_test_f1'][gs.best_index_],
                gs.cv_results_['mean_train_f1'][gs.best_index_],
                gs.cv_results_['std_train_f1'][gs.best_index_]))
       
    print("Recall: {:.2f}, Precision: {:.2f}".format(
            # TPR Sensitivity, hit rate, recall, or true positive rate
            gs.cv_results_['mean_test_recall'][gs.best_index_],
            # PPV Precision or positive predictive value
            gs.cv_results_['mean_test_precision'][gs.best_index_]))
    
    if 'mean_test_roc_auc' in gs.cv_results_.keys():
        print("ROC_AUC: {:.2f}".format(gs.cv_results_['mean_test_roc_auc'][gs.best_index_]))

    print("\nParameters: \n{}\n".format(gs.best_params_))
 