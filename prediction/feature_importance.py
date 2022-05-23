# This is the code that I had for feature analysis 

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
idx = np.arrange(ft.SKB_Flag.sum())

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
