# Questionnaire Analysis 

## dfs folder
Importing datasets as stored in Googlesheets:
- PR-Questionnaire-Raw.csv: Canine Behaviour Asssessment Research Questionnaire (C-BARQ) was answered by Puppy Raisers at week 0 after the start of training. This questionnaire was developed by \cite{Ley2009, Ley2009a, Ley2009b}

- DT-Questionnaire-Raw.csv: Monash Canine Questionnaire - Revised (MCPQ-R) was answered by Dog Trainers at week 10 after the start of Training. This questionnaire was developed by \cite{Serpell2001, Hsu2003, Duffy2012}

- Data Collection - Summary tab: Googlesheet containing a summary of data collection status for each dog.

- Data Collection - Dogs tab: Googlesheet containing demographic data about the dogs participating in the study.

- Data Collection - Training tab: Googlesheet  containing data about the dog's training journey.

## src folder

### dt_main.py
Imports the Questionnaires raw data and "Data Collection - Training" tab, the main changes to the questionnaire raw data are:

- rename columns to more readable column names
- insert column for DOA (Date of Arrival)
- insert column for Status (Training Outcome) - Assistance Dog (AD), Guide Dog (GD) and Withdrawn (W)
- insert column for End Date (Date when dog finished training)

Saves as 'Date-DT_MCPQ-R.csv' and 'Date-PR_C-BARQ.csv' files.


### pr_main.py


## analysis

pr_analysis = Puppy raiser questionnaire analysis (C-BARQ)
dt_analysis = Dog Trainer questionnaire analysis (MCPQ-R)

2_outcomes considers (Fail, Success)
3_outcomes considers (GD = Guide Dog, AD = Assistance Dog, W = Withdraw)

### summary.py