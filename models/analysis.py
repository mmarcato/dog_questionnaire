# gs = joblib.load(os.path.join(dir_model, "dtq-pca-lr.pkl" ))
gs = joblib.load(os.path.join(dir_model, "prq-spl-pca-lr.pkl" ))
# gs = joblib.load(os.path.join(dir_model, "prq-knn-pca-lr.pkl" ))

print('PCA')
print('explained_variance_: ', pca.explained_variance_)
print('explained_variance_ratio_: ', pca.explained_variance_ratio_)
print('noise_variance_: ', pca.noise_variance_)

ft = gs.best_estimator_['ft'].attribute_names[gs.best_estimator_['var'].variances_ > 0]
pca = gs.best_estimator_['pca']
df_pca = pd.DataFrame(pca.components_.transpose()).set_index(ft)
print(df_pca)

pca.singular_values_
pca.n_components_
pca.n_features_
pca.n_samples_

print('LOGISTIC REGRESSION')
lr = gs.best_estimator_['lr']
print('intercept_: ', lr.intercept_)
print('coef_: ', lr.coef_)
lr.n_features_in_
lr.n_iter_


ft = gs.best_estimator_['ft'].attribute_names[gs.best_estimator_['var'].variances_ > 0]
pd.DataFrame(lr.coef_.flatten()).set_index(ft)