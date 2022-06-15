from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FactorAnalysis
from missingpy import MissForest
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2, f_regression

# ------------------------------------------------------------------------- #
#                                   Learning                                #
# ------------------------------------------------------------------------- # 
 
# OTHER METHODS


pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('imp', "passthrough"),
    ('var', VarianceThreshold()),
    ('uni', FeatureUnion([
        ('skb', SelectKBest()),
        ('linear_pca', PCA()), 
        ('kernel_pca', KernelPCA()),
        ('svd', TruncatedSVD()),
        ('fa', FactorAnalysis())
        ])),
    ('skb', 'passthrough'),
    ('clf', RandomForestClassifier(n_jobs = 1, max_features = None, 
                random_state = 0))
    ], memory = memory)
                        
params = {
            'imp': [SimpleImputer(strategy = 'mean'), 
                    SimpleImputer(strategy = 'median'),
                    SimpleImputer(strategy = 'most_frequent'), 
                    KNNImputer(weights = 'uniform'),
                    KNNImputer(weights = 'distance'),
                    MissForest(random_state = 0)],

            'uni__skb__score_func' : [f_classif, chi2],
            'uni__skb__k':  [5, 7, 10, 12, 15],  # 26 features in total

            'uni__linear_pca__n_components': [3,5,7],
            'uni__kernel_pca__n_components': [3,5,7],            
            'uni__scd__n_components': [3,5,7],
            'uni__fa__n_components': [3,5,7],

            'skb':  [10, 20, 30],  # max 43 features

            'clf__max_depth': [2, 3, 5],
            'clf__n_estimators': [25, 50, 75, 100]
        }


pipe2 = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('imp', SimpleImputer(missing_values=np.nan)),
    ('uni', FeatureUnion([
        ('all', DataFrameSelector(feat)),
        ('linear_pca', PCA()), 
        ('kernel_pca', KernelPCA())
        ])),
    ('var', VarianceThreshold()),
    ('slt', SelectKBest()),
    ('clf', RandomForestClassifier(n_jobs = 1, max_features = None, 
                random_state = 0))
    ], memory = memory)

params2 = {
            'uni__linear_pca__n_components': [3,5,7],
            'uni__kernel_pca__n_components': [3,5,7],
            'slt__score_func' : [f_classif, chi2], 
            'slt__k':  [5, 7, 10, 12, 15],  # 26 features in total
            'clf__max_depth': [2, 3, 5],
            'clf__n_estimators': [25, 50, 75, 100]
        }

### THINGS I TRIED
'''
1. IMPUTERS (FOR PRQ ONLY!)
    PARAMS =    
        'imp': [SimpleImputer(strategy = 'mean'), 
            SimpleImputer(strategy = 'median'),
            SimpleImputer(strategy = 'most_frequent'), 
            KNNImputer(weights = 'uniform'),
            KNNImputer(weights = 'distance'),
            MissForest(random_state = 0)]

2. USING STANDARDSCALER
        # ('slc', StandardScaler()),

3. DECOMPOSITION
    PIPE = 
        ('skb', SelectKBest()),
        ('linear_pca', PCA()), 
        ('kernel_pca', KernelPCA()),
        ('svd', TruncatedSVD()),
        ('fa', FactorAnalysis())   

4. LOGISTIC REGRESSION WITH SOLVER = SAGA PENALTY = ELASTICNET, AND L1, L2 RATIO
    PIPE = 
        ('lr', LogisticRegression(solver = 'saga', random_state = 0))
    PARAMS = 
        {'lr__penalty': ['elasticnet'],
        'lr__l1_ratio' : np.linspace(0, 1, 10)}



'''
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
            # 'slt__k':  [5, 10, 15, 20]
}
