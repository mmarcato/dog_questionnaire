import pandas as pd
import numpy as np
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

my_dir = os.path.dirname(os.getcwd())
df_dir = os.path.join(my_dir, "data")

#------------------------------------------------------------------------------------#
#                                     Functions                                      #
#------------------------------------------------------------------------------------#

def import_pr(base_dir):
    # import questionnaire raw data as dataframe and rename columns
    df = pd.read_csv(("%s\\PRQ-Raw.csv" % base_dir), 
          header = 0, parse_dates = ['Timestamp'])

    # clean up the columns names
    # select columns names
    cols = df.columns
    # remove punctuation
    cols = cols.str.replace('[^\w\s]','')
    # delete leading space
    cols = cols.str.strip()
    
    # renaming columns to formated strings
    df.columns = cols
    
    # add suffix
    new_names = [(i,'I_'+i) for i in df.iloc[:, df.columns.str.contains("^\d")].columns.values]
    df.rename(columns = dict(new_names), inplace=True)

    # code column -> rename & set as index 
    df.rename(columns={ df.columns[1]: "Code" }, inplace = True)
    df.set_index('Code', inplace = True)
    # clean up the codes
    df.index = df.index.str.replace(' ','')
    df.index = df.index.str.upper()
    # dropping the two instances I don't know who the dogs are
    df.drop(['4561','CASE/ID#'], axis = 0, inplace = True)

    # drop consent column 
    df.drop([df.columns[1]], axis = 1, inplace = True)

    # replacing values
    df.replace({'Never' : 0, 
                'Seldom' : 1, 
                'Sometimes' : 2, 
                'Usually' : 3, 
                'Always' : 4, 
                'Not observed' : np.nan, 
                'Not observed/Not applicable': np.nan, 
                'Not observed/NA' :np.nan}, 
                    inplace = True)

    # reorder columns to have the text columns at the end
    # columns without describe
    idx1  = df.columns[~df.columns.str.contains('describe')]
    # change datatype to float 
    df[idx1[1:]] = df[idx1[1:]].astype(float)

    # columns with describe
    idx2 = df.columns[df.columns.str.contains('describe')]
    idx1 = idx1.append(idx2)
    df = df[idx1]

    # select questions 5,6,7 and reverse scale
    df.iloc[:,5:8] = np.abs(df.iloc[:,5:8] - 4)

    return df

def score(df):
    return np.where(df.isnull().sum(axis = 1) / df.shape[1] > 0.25, 
                                np.nan , df.mean(axis = 1))

def factor(df):
    df['F_01 Stranger-directed aggression'] = score(df.iloc[:,[10, 11, 12, 15, 16, 18, 20, 21, 22, 28]] )
    df['F_02 Owner-directed aggression'] = score(df.iloc[:,[9, 13, 14, 17, 19, 25, 30, 31]] )
    df['F_03 Dog-directed aggression'] = score(df.iloc[:,[23, 24, 26, 29]] )
    df['F_04 Dog-directed fear'] = score(df.iloc[:,[45, 46, 52, 53]] )
    df['F_05 Dog rivalry'] = score(df.iloc[:,[32, 33, 34, 35]] )
    df['F_06 Trainability'] = score(df.iloc[:,[1, 2, 3, 4, 5, 6, 7, 8]] )
    df['F_07 Chasing'] = score(df.iloc[:,[27, 74, 75, 76]] )
    df['F_08 Stranger-directed fear'] = score(df.iloc[:,[36, 37, 39, 40]] )
    df['F_09 Nonsocial fear'] = score(df.iloc[:,[38, 41, 42, 44, 47, 48]] )
    df['F_10 Separation-related problems'] = score(df.iloc[:,[54, 55, 56, 57, 58, 59, 60, 61]] ) 
    df['F_11 Touch sensitivity'] = score(df.iloc[:,[43, 49, 50, 51]] ) 
    df['F_12 Excitability'] = score(df.iloc[:,[62, 63, 64, 65, 66, 67]] ) 
    df['F_13 Attachment/attention-seeking'] = score(df.iloc[:,[68, 69, 70, 71, 72, 73]] ) 
    df['F_14 Energy'] = score(df.iloc[:,[91, 92]] ) 

    return df

''' Formulas for calculating factors
    The C-BARQ provides a set of quantitative scores for the following fourteen different subscales or categories of behavior:
        'Stranger-directed aggression' score = (questionnaire items 10 + 11 + 12 + 15 + 16 + 18 + 20 + 21 + 22 + 28)/10.
        'Owner-directed aggression' score = (items 9 + 13 + 14 + 17 + 19 + 25 + 30 + 31)/8.
        'Dog-directed aggression' = (items 23 + 24 + 26 + 29)/4
        'Dog-directed fear' = (items 45 + 46 + 52 + 53)/4.
        'Dog rivalry'(familiar dog aggression) score = (items 32 + 33 + 34 + 35)/4
        'Trainability' score = (items 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8)/8—remember to reverse scoring order for items 5, 6 & 7 (see above).
        'Chasing' score = (items 27 + 74 + 75 + 76)/4
        'Stranger-directed fear' score = (items 36 + 37 + 39 + 40)/4
        'Nonsocial fear' score = (items 38 + 41 + 42 + 44 + 47 + 48)/6
        'Separation-related problems' score = (items 54 + 55 + 56 + 57 + 58 + 59 + 60 + 61)/8
        'Touch sensitivity' score = (items 43 + 49 + 50 + 51)/4
        'Excitability' score = (items 62 + 63 + 64 + 65 + 66 + 67)/6
        'Attachment/attention-seeking' score = (items  68 + 69 + 70 + 71 + 72 + 73)/6
        'Energy' score = (items 91 + 92)/2

    If more than 25% of the items in a subscale are missing values, the factor/subscale score should be recorded as a missing value.

    Factors use 1–76 and 91–92
    Factors do not use item 77-90 and 93-100 
'''

def import_dog(base_dir):
    df = pd.read_csv('{}\\Data Collection - Dogs.csv'.format(base_dir), index_col = 0,
            usecols = ['Code', 'Name', 'DOB',  'Sex', 'Breed', 'Source', 'DOA','PR Sup', 'Coat Colour',  'Status', 'End Date'], 
            parse_dates= ['DOA', 'DOB', 'End Date'], dayfirst = True)
    print('Columns in Demographics dataframe: \n', df.columns.tolist())
    print('\nShape of Demographics dataframe: \n', df.shape)
    # calculating duration of training
    df['Duration'] = df['End Date'] - df['DOA']
    # defining training outcome
    df.Status.replace({"CD" : "AD"}, inplace = True)
    df['Outcome'] = np.select( [df['Status'] == 'in Training',
                            df['Status'] == 'W', 
                            df['Status'] == 'GD', 
                            df['Status'] == 'AD'], 
                            [np.nan, 'Fail', 'Success', 'Success'])
    df['Label'] = np.select([df['Status'] == 'in Training',
                            df['Status'] == 'W', 
                            df['Status'] == 'AD',
                            df['Status'] == 'GD'], 
                            [np.nan, 0, 1,  2])
    return df

#------------------------------------------------------------------------------------#
#                                        Main                                        #
#------------------------------------------------------------------------------------#

df_pr =  import_pr(df_dir)   
df_pr = factor(df_pr)
df_dogs = import_dog(df_dir)

# merging dataframes based on the code, to add columns for status, DOA and end date
df = df_pr.merge(df_dogs, left_index = True, right_index = True, how = 'left')

print('\n\nNumber of dogs with two answers for the PR Questionnaire:', df.index.duplicated().sum())
print('Dogs with duplicated answers: ', df.loc[df.index.duplicated() == True, 'Name'])
print('The first instance of a duplicate was DROPPED\nThe last instance of a duplicate was KEPT')
df.drop_duplicates('Name', keep = 'last', inplace = True)
print('\n\nDataset size: ', df.shape)


# save the dataframe
df.to_csv('%s\\2022-08-08-PRQ_C-BARQ.csv' % df_dir, index_label = 'Code')
print('New PRQ_C-BARQ Dataset available at: ', df_dir)
