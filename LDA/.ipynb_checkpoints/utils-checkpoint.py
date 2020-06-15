# from parameters import *
import pandas as pd
import os
# import numpy as np

######################################
# -----------Read data----------------#
######################################


def read_data(in_dir, filename, text_col, renamed_text_col, text_language = 'EN' ):
    assert len(renamed_text_col) > 0
    df = pd.read_csv(os.path.join(in_dir, filename))
    df.rename(columns={text_col: renamed_text_col}, inplace=True)

    if text_language == 'EN':
        # filter english only and not null rows
        df = df.loc[(~df[renamed_text_col].isna()) & (df['Language answered'] == 'EN'),
                    [renamed_text_col]]
    elif text_language == 'FR':
        df = df.loc[(~df[renamed_text_col].isna()) & (df['Language answered'] == 'FR'),
                    [renamed_text_col]]
    elif text_language == 'ALL':
        df = df.loc[~df[renamed_text_col].isna(), [renamed_text_col]]

    else:
        raise ValueError

    df = df.reset_index()

    return df


# keep some columns for analysis part
cols = ['Quarter','Submitted date','What was the main purpose of your visit today? Please choose one option only. The list represents broad categories, and you may be asked to further specify the main reason why you visited today.','Approximately how often do you visit My Account?',
 "Overall how satisfied were you with today's experience?", 'Which option did you choose to login to My Account?', 'Did/do you need to call the CRA to complete your visit/transaction today?',
 'Are you:', 'I live in...' ,
 'What is the highest level of education you have completed?', 'Would you be willing to indicate in which of the following age categories you belong?',
'What language do you speak most often at home?', 'Which of the following categories best describes your current employment status?', 'Which of the following categories best represents your most recent annual household income, before taxes?', 
'Are you an Aboriginal person?', 'Are you a person with a disability?', 'Are you a member of a visible minority group?', 'From what type of device did you access My Account today?']

def read_data_meta(in_dir, filename, text_col, renamed_text_col, text_language = 'EN' , other_cols = cols):
    assert len(renamed_text_col) > 0
    df = pd.read_csv(os.path.join(in_dir, filename))
    df.rename(columns={text_col: renamed_text_col}, inplace=True)
    columns = [renamed_text_col] + cols
    if text_language == 'EN':
        # filter english only and not null rows
        df = df.loc[(~df[renamed_text_col].isna()) & (df['Language answered'] == 'EN'),
                    columns]
#     elif text_language == 'FR':
#         df = df.loc[(~df[renamed_text_col].isna()) & (df['Language answered'] == 'FR'),
#                     [renamed_text_col]]
#     elif text_language == 'ALL':
#         df = df.loc[~df[renamed_text_col].isna(), [renamed_text_col]]

    else:
        raise ValueError

    df = df.reset_index()

    return df