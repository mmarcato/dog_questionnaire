
# new pipeline

pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('uni', FeatureUnion([
        ('all', DataFrameSelector(feat)),
        ('linear_pca', PCA()), 
        ('kernel_pca', KernelPCA())
        ])),
    ('var', VarianceThreshold()),
    ('slt', SelectKBest()),
    ('clf', RandomForestClassifier(n_jobs = 1, max_features = None, 
                random_state = 0))
    ])


pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('var', VarianceThreshold()),
    ('uni', FeatureUnion([
        ('skb', SelectKBest()),
        ('linear_pca', PCA()), 
        ('kernel_pca', KernelPCA())
        ])),
    ('clf', RandomForestClassifier(n_jobs = 1, max_features = None, 
                random_state = 0))
    ])


                        
params = {
            'uni__linear_pca__n_components': [3,5,7],
            'uni__kernel_pca__n_components': [3,5,7],
            'slt__score_func' : [f_classif, chi2], 
            'slt__k':  [5, 7, 10, 12, 15],  # 26 features in total
            'clf__max_depth': [2, 3, 5],
            'clf__n_estimators': [25, 50, 75, 100]
        }
