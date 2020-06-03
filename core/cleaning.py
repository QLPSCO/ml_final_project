
from datetime import datetime
import pandas as pd
import numpy as np
import glob
from imblearn.under_sampling import RandomUnderSampler

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
                            #'EMPLOYER_NAME': 'category', # previous year's result
                            'EMPLOYER_STATE': 'category',
                            'NAICS_CODE': 'category',
                            'AGENT_REPRESENTING_EMPLOYER': 'boolean',
                            'TOTAL_WORKER_POSITIONS': 'int',
                            'WAGE_RATE_OF_PAY_FROM_1': 'float',
                            'H-1B_DEPENDENT': 'boolean',
                            'WILLFUL_VIOLATOR': 'boolean',
                            'NEW_EMPLOYMENT': 'ratio',
                            'CONTINUED_EMPLOYMENT': 'ratio',
                            'CHANGE_PREVIOUS_EMPLOYMENT': 'ratio',
                            'NEW_CONCURRENT_EMPLOYMENT': 'ratio',
                            'CHANGE_EMPLOYER': 'ratio',
                            'AMENDED_PETITION': 'ratio',
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
    print(4, datetime.now().strftime("%H:%M:%S"))

    # Remove applications that were withdrawn
    df = df[df['CASE_STATUS'] != 'WITHDRAWN'].reset_index()
    print(4.2, datetime.now().strftime("%H:%M:%S"))

    # Reclassify applications that were withdrawn after being certified as simply certified
    df.loc[df['CASE_STATUS'] == 'CERTIFIED-WITHDRAWN', 'CASE_STATUS'] = 'CERTIFIED'
    print(4.5, datetime.now().strftime("%H:%M:%S"))

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

    # Clean up the SOCS code
    regex = '\d\d-\d\d\d\d\.\d\d'
    df.loc[df['SOC_CODE'].str.contains(regex).fillna(False), 'SOC_CODE'] = \
    df.loc[df['SOC_CODE'].str.contains(regex).fillna(False), 'SOC_CODE'].str[:7]

    november_fixer = {'Nov-11': '11-2011',
                      'Nov-21': '11-3021',
                      'Nov-22': '11-2022',
                      'Nov-31': '11-3031',
                      'Nov-51': '11-3051',
                      'Nov-61': '11-3061',
                      'Nov-71': '11-3071',
                      'Nov-13': '11-3013',
                      'Nov-32': '11-2032',
                      'Nov-33': '11-9033',
                      'Nov-39': '11-9039',
                      'Nov-41': '11-9041',
                      'Nov-81': '11-9081',
                      'Nov-99': '11-9199'}
    df['SOC_CODE'] = df['SOC_CODE'].replace(november_fixer)

    regex = '^\D+$'
    df.loc[df['SOC_CODE'].str.contains(regex).fillna(False), 'SOC_CODE'] = 'Not specified'
    df['SOC_CODE'] = df['SOC_CODE'].fillna('Not specified')

    # Condense NAICS code to be four-digit level
    df['NAICS_CODE'] = df.NAICS_CODE.astype(str).str[:4]

    # Convert employment status subcats to be proportions rather than raws
    applicant_count = pd.Series(0, index=np.arange(len(df)))
    for col in [k for k, v in columns_for_analysis.items() if v == 'ratio']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        applicant_count += df[col].fillna(0)

    for col in [k for k, v in columns_for_analysis.items() if v == 'ratio']:
        df[col] = df[col].fillna(-1) / applicant_count
        if min(df[col]) < 0:
            df.loc[df[col] < 0, col] = df[col][df[col] >= 0].median()
        df[col] = df[col][np.isinf(df[col])] = 0 # for nonzero / 0
        df[col] = df[col].fillna(0) # for 0 / 0

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
    df = df.drop(columns=['PERIOD_OF_EMPLOYMENT_START_DATE',
                          'PERIOD_OF_EMPLOYMENT_END_DATE'])
    print(12, datetime.now().strftime("%H:%M:%S"))

    # Number of applications submitted by company
    employer_count = pd.DataFrame(df['EMPLOYER_NAME'].value_counts()) # break out by year!
    employer_count = employer_count.rename(columns={'EMPLOYER_NAME': 'TOTAL_ANNUAL_APPLICATIONS_BY_EMPLOYER'})
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

def undersample(dataframe):
    df = dataframe.copy()

    rus = RandomUnderSampler(random_state=20170217)

    x, y = rus.fit_resample(df.drop(columns='CASE_STATUS'), df['CASE_STATUS'])

    x['CASE_STATUS'] = y

    return x
