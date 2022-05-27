import matplotlib.pyplot as plt  
import seaborn as sns

# correlation between features
fig, ax = plt.subplots(figsize=(20,10))         
corr = X.corr()
sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=14)
plt.show()

# ------------------------------------------------------------------------- #
#                  Feature Importance  (Simple Pipeline)                    #
# ------------------------------------------------------------------------- #

print('Evaluate Grid Search output\n')
gs =  joblib.load(os.path.join(dir_model, "inner-2_outcomes.pkl" ))
gs_output(gs)
print(gs.best_estimator_)

# feature importances dataframe
ft = pd.DataFrame({'Name' : gs.best_estimator_['ft'].attribute_names})

# Above Variance Threshold
ft['Var_Flag'] = False
ft.loc[gs.best_estimator_['var'].variances_ > 0, 'Var_Flag'] = True

# Select K Best - Scores
ft['SKB_Scores'] = np.nan
ft.loc[ft['Var_Flag'],'SKB_Scores'] = gs.best_estimator_['slt'].scores_

# creating a flag for features that were selected by skb
ft['SKB_Flag'] = False
ft.loc[ft.nlargest(gs.best_params_['slt__k'],['SKB_Scores']).index, 'SKB_Flag'] = True

# Random Forest
ft['RF_Importances'] = np.nan
ft.loc[ft.SKB_Flag == 1, 'RF_Importances'] = gs.best_estimator_['clf'].feature_importances_

# ------------------------------------------------------------------------- #
#                         Plot Feature Importance                           #
# ------------------------------------------------------------------------- #

# plot feature importances
plt.figure(figsize=(12,5))
width = 0.3
idx = np.arange(ft.SKB_Flag.sum())

# SKB feature scores for those selected by SKB
skb_norm = ft.loc[ft.SKB_Flag == True, "SKB_Scores"] / ft.loc[ft.SKB_Flag == True, "SKB_Scores"].sum()
plt.bar(idx, skb_norm, width, label = "SKB Scores")
# RF feature importances for those selected by SKB
plt.bar(idx + width, ft.loc[ft.SKB_Flag == True, "RF_Importances"], width, label='RF Importances')
plt.xticks(rotation = 45)

plt.xlabel('Features - items and factors numbers')
plt.ylabel('Normalised values')
plt.title('SKB Scores and RF Importances for the {} highest scoring features'.format(ft.SKB_Flag.sum()))
plt.xticks(idx + width / 2, ft.loc[ft.SKB_Flag == True, "Name"].str.split().str[0])

# Place legend in the best position 
plt.legend(loc='best')
plt.show()
# ------------------------------------------------------------------------- #
#                  Feature Importance  (Complex Pipeline)                   #
# ------------------------------------------------------------------------- #

print('Evaluate Grid Search output\n')
gs =  joblib.load(os.path.join(dir_model, "dt-2_outcomes-1.pkl" ))
gs_output(gs)
print(gs.best_estimator_)

# feature importances dataframe
ft = pd.DataFrame({'Name' : gs.best_estimator_['ft'].attribute_names})

# Above Variance Threshold
ft['Var_Flag'] = False
ft.loc[gs.best_estimator_['var'].variances_ > 0, 'Var_Flag'] = True

# Select K Best - Scores
ft['SKB_Scores'] = np.nan
ft.loc[ft['Var_Flag'] == True,'SKB_Scores'] = gs.best_estimator_.named_steps["uni"].transformer_list[0][1].scores_

# creating a flag for features that were selected by skb
ft['SKB_Flag'] = False
ft.loc[ft.nlargest(gs.best_params_['uni__skb__k'],['SKB_Scores']).index, 'SKB_Flag'] = True

# Linear & Kernel PCA
linear_pca = gs.best_estimator_.named_steps["uni"].transformer_list[1][1].singular_values_
kernel_pca = gs.best_estimator_.named_steps["uni"].transformer_list[2][1].eigenvalues_

linear_vars = ['linear_' + str(i) for i in list(range(len(linear_pca)))]
kernel_vars = ['kernel_' + str(i) for i in list(range(len(linear_pca)))]

pca = pd.DataFrame({ "Name": np.append(linear_vars, kernel_vars), 
                "PCA" : np.append(linear_pca, kernel_pca)})
# concatenates the two dataframes
ft = pd.concat([ft, pca], ignore_index = True)

# Random Forest
ft['RF_Importances'] = np.nan
# selects feature from skb and the pca (kernel, linear)
ft.loc[(ft.SKB_Flag == True) | ~ft.PCA.isna(), 'RF_Importances'] = gs.best_estimator_['clf'].feature_importances_


# ------------------------------------------------------------------------- #
#                         Plot Feature Importance                           #
# ------------------------------------------------------------------------- #

# plot feature importances
plt.figure(figsize=(12,5))
labels = ft.loc[ft.SKB_Flag == True, "Name"].str.split().str[0].append(ft.loc[~ft.PCA.isna(), "Name"], ignore_index = True)
idx = labels.index
width = 0.3
skb_nan = pd.Series([0]).repeat(ft.PCA.count())


# SKB feature scores for those selected by SKB
skb_norm = ft.loc[ft.SKB_Flag == True, "SKB_Scores"] / ft.loc[ft.SKB_Flag == True, "SKB_Scores"].sum()

plt.bar(idx, skb_norm.append(skb_nan), width, label = "SKB Scores")
# RF feature importances for those selected by SKB
plt.bar(idx + width, ft.loc[~ft.RF_Importances.isna(), "RF_Importances"], width, label='RF Importances')
plt.xticks(rotation = 45)

plt.xlabel('Features - items and factors numbers')
plt.ylabel('Normalised values')
plt.title('SKB Scores {} highest scoring features and RF Importances for SKB, PCA Linear and Kernel'.format(ft.SKB_Flag.sum()))
plt.xticks(idx + width / 2, labels)

# Place legend in the best position 
plt.legend(loc='best')
plt.show()



from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion

# load gs results from pickle file
gs_load =  joblib.load(os.path.join(dir_model, "pr-2_outcomes-1.pkl" ))
print('Evaluate Grid Search output\n')
gs_output(gs_load)
gs = gs_load


# code from https://gist.github.com/nbertagnolli/bf5bd2cc7e0b142d6a862e54dd3ac871

# I got this on the internet and it didn't work!
from typing import List, Tuple
def extract_feature_names(model, name) -> List[str]:
    """Extracts the feature names from arbitrary sklearn models
    
    Args:
        model: The Sklearn model, transformer, clustering algorithm, etc. which we want to get named features for.
        name: The name of the current step in the pipeline we are at.
        
    Returns:
        The list of feature names.  If the model does not have named features it constructs feature names
        by appending an index to the provided name.
    """
    if hasattr(model, "get_feature_names"):
        return model.get_feature_names()
    elif hasattr(model, "n_clusters"):
        return [f"{name}_{x}" for x in range(model.n_clusters)]
    elif hasattr(model, "n_components"):
        return [f"{name}_{x}" for x in range(model.n_components)]
    elif hasattr(model, "components_"):
        n_components = model.components_.shape[0]
        return [f"{name}_{x}" for x in range(n_components)]
    elif hasattr(model, "classes_"):
        return model.classes_
    elif hasattr(model, "n_features_in_"):
        return model.n_features_in_
    else:
        return [name]

def get_feature_names(model, names: List[str], name: str) -> List[str]:
    """Thie method extracts the feature names in order from a Sklearn Pipeline
    
    This method only works with composed Pipelines and FeatureUnions.  It will
    pull out all names using DFS from a model.
    
    Args:
        model: The model we are interested in
        names: The list of names of final featurizaiton steps
        name: The current name of the step we want to evaluate.
    
    Returns:
        feature_names: The list of feature names extracted from the pipeline.
    """
    
    # Check if the name is one of our feature steps.  This is the base case.
    print(name)
    print(names)
    if name in names:
        # If it has the named_steps atribute it's a pipeline and we need to access the features
        if hasattr(model, "named_steps"):
            # FIXME:: NEED BASE CASE
            return extract_feature_names(model.named_steps[name], name)
        # Otherwise get the feature directly
        else:
            return extract_feature_names(model, name)
    elif type(model) is Pipeline:
        feature_names = []
        for name in model.named_steps.keys():
            feature_names += get_feature_names(model.named_steps[name], names, name)
        return feature_names
    elif type(model) is FeatureUnion:
        feature_names= []
        for name, new_model in model.transformer_list:
            feature_names += get_feature_names(new_model, names, name)
        return feature_names
    # If it is none of the above do not add it.
    else:
        return []

get_feature_names(gs.best_estimator_, ["uni", "slt", "linear_pca", "kernel_pca"], None)

get_feature_names(gs.best_estimator_, ["slt"], None)


for name, new_model in gs.best_estimator_.named_steps["uni"].transformer_list:
    print(name, new_model)
    if type(new_model) is SelectKBest:
        print(len(new_model.scores_))





# This is the code that I had for feature analysis 
