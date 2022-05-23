''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
# ------------------------------------------------------------------------- #
#                           Importing Global Modules                        #
# ------------------------------------------------------------------------- #
import os
import numpy as np
import pandas as pd
from time import time

# ------------------------------------------------------------------------- #
#                             Local Imports                                 #    
# ------------------------------------------------------------------------- # 

# Define local directories
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_base = os.path.dirname(dir_current)

# directory where the dataset is located
dir_df = os.path.join(dir_base, 'data')

# ------------------------------------------------------------------------- #
#                           Import Dasets from Folder                       #
# ------------------------------------------------------------------------- #

dtq = pd.read_csv(os.path.join(dir_df, '2022-05-10-DT_MCPQ-R.csv'))
prq = pd.read_csv(os.path.join(dir_df, '2022-05-10-PR_C-BARQ.csv'))

### 1. dataset union = outer
df_outer = pd.merge(prq.set_index('Code'), dtq.set_index('Code'), 
                how = "outer", left_index = True, right_index = True, 
                suffixes = ("_x", None))

duplicates = df_outer.columns[df_outer.columns.str.contains('_x')]
for col in duplicates:
    print(col, df_outer[col[:-2]].isna().sum())
    df_outer[col[:-2]].fillna(df_outer[col], inplace = True)
df_outer.drop(duplicates, axis = 1, inplace = True)

### 2. dataset intersection = inner
df_inner = pd.merge(prq.set_index('Code'), dtq.set_index('Code'),
                how = "inner", left_index = True, right_index = True, 
                suffixes = ("_x", None))

df_inner.drop(df_inner.columns[df_inner.columns.str.contains('_x')]
                , axis = 1, inplace = True)

df_outer.to_csv(os.path.join(dir_df, '2022-05-13-df-outer.csv'))
df_inner.to_csv(os.path.join(dir_df, '2022-05-13-df-inner.csv'))

# OLD STUFF
# CHECKING IF ALL QUESTIONNAIRES WERE FOUND CONSIDERING SUMMARY GOOGLE SHEET TAB 
df_sm = pd.read_csv(("%s\\Data Collection - Summary.csv" % my_dir), 
    usecols = [2],  skiprows=1)

df_dt = pd.read_csv(("%s\\DT-Questionnaire-Raw.csv" % my_dir), 
    usecols = [1], skiprows= 1,
    header = None, names = ['dt_quest'])
df = pd.merge(df_sm, df_dt, left_on = 'Code', right_on = 'dt_quest', how = 'left')

if np.NaN
df['DT-Quest'] = df.loc[df['dt_quest'] =! pd.notna

set(set(df_dt['dt_quest']) - set(df_sm['Code']))
set(set(df_sm['Code']) - set(df_dt['dt_quest']))