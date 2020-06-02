
from datetime import datetime
import pandas as pd
import numpy as np
import glob

def preprocess_data(filepath):
    '''
    Completes preprocessing specific to the LCA dataset.

    Input:
      - filepath (string): folder where all CSVs are located
    Returns:
      - pickled dataframe in the same location
    '''
    print(1, datetime.now().strftime("%H:%M:%S"))

    # df = dataframe.copy()
    year_mismatches = {'H1B_DEPENDENT': 'H-1B_DEPENDENT',
                       'EMPLOYMENT_START_DATE': 'PERIOD_OF_EMPLOYMENT_END_DATE',
                       'EMPLOYMENT_END_DATE': 'PERIOD_OF_EMPLOYMENT_START_DATE',
                       'WAGE_RATE_OF_PAY_FROM': 'WAGE_RATE_OF_PAY_FROM_1',
                       'TOTAL_WORKERS': 'TOTAL_WORKER_POSITIONS',
                       'NEW_CONCURRENT_EMP': 'NEW_CONCURRENT_EMPLOYMENT'}

    columns_for_analysis = {'CASE_STATUS': 'category',
                            'CASE_SUBMITTED': 'datetime64',
                            'SOC_CODE': 'category',
                            'FULL_TIME_POSITION': 'boolean',
                            'PERIOD_OF_EMPLOYMENT_START_DATE': 'datetime64',
                            'PERIOD_OF_EMPLOYMENT_END_DATE': 'datetime64',
                            'EMPLOYER_NAME': 'category',
                            'EMPLOYER_STATE': 'category',
                            'NAICS_CODE': 'category',
                            'AGENT_REPRESENTING_EMPLOYER': 'boolean',
                            'TOTAL_WORKER_POSITIONS': 'int',
                            'WAGE_RATE_OF_PAY_FROM_1': 'float',
                            'H-1B_DEPENDENT': 'boolean',
                            'WILLFUL_VIOLATOR': 'boolean',
                            'NEW_EMPLOYMENT': 'boolean',
                            'CONTINUED_EMPLOYMENT': 'boolean',
                            'CHANGE_PREVIOUS_EMPLOYMENT': 'boolean',
                            'NEW_CONCURRENT_EMPLOYMENT': 'boolean',
                            'CHANGE_EMPLOYER': 'boolean',
                            'AMENDED_PETITION': 'boolean',
                            'YEAR': 'category'}
    print(2, datetime.now().strftime("%H:%M:%S"))

    df = pd.DataFrame()
    for year in glob.glob(filepath + '/*.csv'):
        print(year)
        data_for_year = pd.read_csv(year, low_memory=False)
        print(3.1, datetime.now().strftime("%H:%M:%S"))

        # rename columns with naming discrepancies across years
        data_for_year = data_for_year.rename(columns=year_mismatches)
        data_for_year['YEAR'] = year[-8:-4]
        print(3.2, datetime.now().strftime("%H:%M:%S"))

        df = df.append(data_for_year, ignore_index=True)
        print(3.3, datetime.now().strftime("%H:%M:%S"))

    # Remove applications for speciality visas for Australia, Chile, & Singapore
    df = df[df['VISA_CLASS'] == 'H-1B']

    # Remove applications that were withdrawn
    df = df[df['CASE_STATUS'] != 'WITHDRAWN']
    print(4, datetime.now().strftime("%H:%M:%S"))

    # Drop any columns not identified for analysis
    df = df.filter(list(columns_for_analysis.keys()))
    print(5, datetime.now().strftime("%H:%M:%S"))

    # remove leading/trailing whitespace from text fields
    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    print(6, datetime.now().strftime("%H:%M:%S"))

    # convert date columns to datetime; retain only date, drop the time
    for col in [k for k, v in columns_for_analysis.items() if v == 'datetime64']:
        df[col] = pd.to_datetime(df[col]).dt.date
    print(7, datetime.now().strftime("%H:%M:%S"))

    # remove punctuation from wage column
    df['WAGE_RATE_OF_PAY_FROM_1'] = (df['WAGE_RATE_OF_PAY_FROM_1']
                                    .str.replace(',|\$', '').astype('float'))
    print(8, datetime.now().strftime("%H:%M:%S"))

    # recode Boolean to be 0/1/NAN
    for col in [k for k, v in columns_for_analysis.items() if v == 'boolean']:
        df[col] = df[col].map({'Y': 1, 'N': 0, 1: 1, 0: 0, '1': 1, '0': 0})
    print(9, datetime.now().strftime("%H:%M:%S"))

    # some columns make sense to have NaNs be recoded as 0, so do that
    nas_to_0_vars = ['H-1B_DEPENDENT', 'WILLFUL_VIOLATOR']
    for col in nas_to_0_vars:
        df.loc[df[col].isna(), col] = 0
    print(10, datetime.now().strftime("%H:%M:%S"))

    # replace NaN's for numerical columns with median values
    for col in [k for k, v in columns_for_analysis.items() if v == 'int' or v == 'float']:
        median = df[col].median()
        df.loc[df[col].isna(), col] = median
    print(11, datetime.now().strftime("%H:%M:%S"))

    # NEW FEATURE GENERATION
    # Duration of employment
    df['EMPLOYMENT_LENGTH'] = ((df['PERIOD_OF_EMPLOYMENT_START_DATE']
                                - df['PERIOD_OF_EMPLOYMENT_END_DATE'])
                                / np.timedelta64(1,'D'))
    print(12, datetime.now().strftime("%H:%M:%S"))

    # Number of applications submitted by company
    employer_count = pd.DataFrame(df['EMPLOYER_NAME'].value_counts())
    employer_count = employer_count.rename(columns={'EMPLOYER_NAME': 'COUNT_BY_EMPLOYER'})
    df = df.merge(employer_count,
            left_on='EMPLOYER_NAME',
            right_index=True,
            validate='m:1')
    print(13, datetime.now().strftime("%H:%M:%S"))

    # Count of missing or invalid fields
    df['NUMBER_INVALID_FIELDS'] = df.isnull().sum(axis=1)
    print(14, datetime.now().strftime("%H:%M:%S"))

    # FINISHED PREPROCESSING
    return df
