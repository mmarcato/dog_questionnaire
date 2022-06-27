import pandas as pd
import numpy as np
import os
my_dir = os.path.dirname(os.getcwd())
df_dir = os.path.join(my_dir, "data")

#------------------------------------------------------------------------------------#
#                                     Functions                                      #
#------------------------------------------------------------------------------------#

def import_dt(base_dir):
    # new column names as the original were quite long
    cols = ['Timestamp', 'Code', 'Consent', 'Extraversion-Active', 'Extraversion-Energetic', 'Extraversion-Excitable', 
    'Extraversion-Hyperactive', 'Extraversion-Lively', 'Extraversion-Restless', 
    'Extraversion-Comments',
    'Motivation-Assertive', 'Motivation-Determined', 'Motivation-Independent', 
    'Motivation-Persevering', 'Motivation-Tenacious', 'Motivation-Comments',
    'Training-Attentive', 'Training-Biddable', 'Training-Intelligent', 'Training-Obedient',
    'Training-Reliable', 'Training-Trainable', 'Training-Comments', 'Amicability-Easy-going', 
    'Amicability-Friendly', 'Amicability-Non-aggressive', 'Amicability-Relaxed', 'Amicability-Sociable', 
    'Amicability-Comments', 
    'Neuroticism-Fearful',   'Neuroticism-Nervous', 'Neuroticism-Submissive', 'Neuroticism-Timid', 'Neuroticism-Comments']
    # import questionnaire raw data as dataframe and rename columns
    df = pd.read_csv(("%s\\DTQ-Raw.csv" % base_dir), names = cols, header = 0, parse_dates = ['Timestamp'])
    # setting the code as new index to make merge easier
    
    df.set_index('Code', inplace = True)
    df.index = df.index.str.strip()

    # droping the timestamp and consent columns 
    df.drop(['Consent'], axis = 1, inplace = True)
    
    return df


def import_dog(base_dir):
    df = pd.read_csv('{}\\Data Collection - Dogs.csv'.format(base_dir), 
            index_col = 0, parse_dates= ['DOA', 'DOB', 'End Date'], dayfirst = True, 
            usecols = ['Code', 'Name', 'DOB',  'Sex', 'Breed', 'Source', 
                    'DOA','PR Sup', 'Coat Colour',  'Status', 'End Date'])

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


def import_training(base_dir):
    # import training data with second row as a header
    df = pd.read_excel('%s\\Data Collection.xlsx' % base_dir, 
            header = [1], index_col = 1, sheet_name= ['Training'])
    # drop columns with nas only and return
    return df['Training'].dropna(axis = 0, how = 'all')


#------------------------------------------------------------------------------------#
#                                        Main                                        #
#------------------------------------------------------------------------------------#

df_dt =  import_dt(df_dir)   
df_training = import_training(df_dir)

# merging dataframes based on the code, to add columns for status, DOA and end date
df = df_dt.merge(df_training[['Name', 'Status', 'DOA', 'End Date']], 
                    left_index = True, right_index = True, how = 'left')

df_dogs = import_dog(df_dir)
# merging dataframes based on the code, to add columns for status, DOA and end date
df = df_dt.merge(df_dogs, left_index = True, right_index = True, how = 'left')


# duplicate statistics and handling
print('\n\nNumber of dogs with at least one answer for the DT Questionnaire:', len(df.index.unique()))
print('\n\nNumber of dogs with two answers for the DT Questionnaire:', df.index.duplicated().sum())
print('Dogs with duplicated answers\n\n', df.loc[df.index.duplicated() == True, 'Name'])
df.drop_duplicates('Name', keep = 'last', inplace = True)
print('\nThe first instance of a duplicate was DROPPED')
print('The last instance of a duplicate was KEPT')

# dataset and training outcome statistics 
print('\n\nDataset size: ', df.shape)
print('Status: \n{}'.format(df.Status.value_counts()))

# save the dataframe
df.to_csv('%s\\2022-06-27-DTQ_MCPQ-R.csv' % df_dir)
print('New DTQ_MCPQ-R Dataset available at: ', df_dir)