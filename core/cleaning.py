
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import random
from imblearn.under_sampling import RandomUnderSampler

def preprocess_data(filepath, nrows=None, skiprows=None):
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
                       'EMPLOYMENT_START_DATE': 'PERIOD_OF_EMPLOYMENT_START_DATE',
                       'EMPLOYMENT_END_DATE': 'PERIOD_OF_EMPLOYMENT_END_DATE',
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
                            # 'NAICS_CODE': 'category',
                            'AGENT_REPRESENTING_EMPLOYER': 'boolean',
                            'TOTAL_WORKER_POSITIONS': 'int',
                            'WAGE_RATE_OF_PAY_FROM_1': 'float',
                            'WAGE_UNIT_OF_PAY_1': 'category',
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
    for year in glob.glob(filepath + '/*.pkl'): #'/*.csv'):
        print(year)
        # data_for_year = pd.read_csv(year, low_memory=False, nrows=nrows, skiprows=skiprows)
        data_for_year = pd.read_pickle(year)
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
    print(5, datetime.now().strftime("%H:%M:%S"))



    # Reclassify applications that were withdrawn after being certified as simply certified
    df.loc[df['CASE_STATUS'] == 'CERTIFIED-WITHDRAWN', 'CASE_STATUS'] = 'CERTIFIED'
    print(6, datetime.now().strftime("%H:%M:%S"))

    # Drop any columns not identified for analysis
    df = df.filter(list(columns_for_analysis.keys()))
    print(7, datetime.now().strftime("%H:%M:%S"))

    # Count of missing or invalid fields
    df['NUMBER_INVALID_FIELDS'] = df.isnull().sum(axis=1)
    print(8, datetime.now().strftime("%H:%M:%S"))

    # remove leading/trailing whitespace from text fields
    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    print(9, datetime.now().strftime("%H:%M:%S"))

    # convert date columns to datetime; retain only date, drop the time
    # for col in [k for k, v in columns_for_analysis.items() if v == 'datetime64']:
    #     df[col] = pd.to_datetime(df[col]).dt.date
    # print(10, datetime.now().strftime("%H:%M:%S"))

    # remove punctuation from wage column
    df['WAGE_RATE_OF_PAY_FROM_1'] = (df['WAGE_RATE_OF_PAY_FROM_1']
                                    .str.replace(',|\$', '').astype('float'))
    print(11, datetime.now().strftime("%H:%M:%S"))
    df['WAGE_RATE_OF_PAY_FROM_1'] = df['WAGE_RATE_OF_PAY_FROM_1'].fillna('Year')

    wage_multiplier = {'Year': 1,
                       'Hour': 2080,
                       'Month': 12,
                       'Week': 52,
                       'Bi-Weekly': 26}

    df['WAGE_RATE_OF_PAY_FROM_1'] = (df['WAGE_RATE_OF_PAY_FROM_1']
                                     * df['WAGE_UNIT_OF_PAY_1'].map(wage_multiplier))

    # Clean up the SOCS code
    # regex = '\d\d-\d\d\d\d\.\d\d'
    # df.loc[df['SOC_CODE'].str.contains(regex).fillna(False), 'SOC_CODE'] = \
    # df.loc[df['SOC_CODE'].str.contains(regex).fillna(False), 'SOC_CODE'].str[:7]

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
    df['SOC_CODE'] = df['SOC_CODE'].astype(str).str[:2]

    regex = '^\D+$'
    df.loc[df['SOC_CODE'].str.contains(regex).fillna(False), 'SOC_CODE'] = np.nan
    print(12, datetime.now().strftime("%H:%M:%S"))

    # Condense NAICS code to be four-digit level
    # df['NAICS_CODE'] = df['NAICS_CODE'].astype(str).str[:2]
    # df['NAICS_CODE'][df['NAICS_CODE'] == 'na'] = np.nan
    # print(13, datetime.now().strftime("%H:%M:%S"))

    # Convert employment status subcats to be proportions rather than raws
    applicant_count = pd.Series(0, index=np.arange(len(df)))
    for col in [k for k, v in columns_for_analysis.items() if v == 'ratio']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        print(sum(df[col].fillna(0)))
        applicant_count += df[col].fillna(0)

    for col in [k for k, v in columns_for_analysis.items() if v == 'ratio']:
        df[col] = (df[col].fillna(-1) / applicant_count)
        if df[col].min() < 0.0:
            df.loc[df[col] < 0, col] = df[col][df[col] >= 0].median()
        df[col][np.isinf(df[col])] = 0 # for nonzero / 0
        df[col] = df[col].fillna(0) # for 0 / 0
    print(14, datetime.now().strftime("%H:%M:%S"))

    # recode Boolean to be 0/1, and replace NaN's weighted randomly
    for col in [k for k, v in columns_for_analysis.items() if v == 'boolean']:
        df[col] = df[col].map({'Y': 1, 'N': 0, 1: 1, 0: 0, '1': 1, '0': 0})
        if len(df[col][df[col].isna()]) > 0:
            val = np.ravel(df[col].values)
            val = val[~pd.isnull(val)]
            val = np.random.choice(val, size=len(df[col][df[col].isna()]))
            df[col].update(pd.Series(val, index=df[col][df[col].isna()].index))
    print(15, datetime.now().strftime("%H:%M:%S"))

    # replace categorical NaN's weighted randomly
    for col in [k for k, v in columns_for_analysis.items() if v == 'category']:
        if len(df[col][df[col].isna()]) > 0:
            val = np.ravel(df[col].values)
            val = val[~pd.isnull(val)]
            val = np.random.choice(val, size=len(df[col][df[col].isna()]))
            df[col].update(pd.Series(val, index=df[col][df[col].isna()].index))
    print(16, datetime.now().strftime("%H:%M:%S"))

    # replace NaN's for numerical columns with median values
    for col in [k for k, v in columns_for_analysis.items() if v == 'int' or v == 'float']:
        median = df[col].median()
        df.loc[df[col].isna(), col] = median
    print(17, datetime.now().strftime("%H:%M:%S"))

    # NEW FEATURE GENERATION
    # number of days before start date the application was submitted
    # df['CASE_SUBMITTED'] = (df['PERIOD_OF_EMPLOYMENT_START_DATE'] - df['CASE_SUBMITTED']).dt.days
    # df.loc[df['CASE_SUBMITTED'].isna(), 'CASE_SUBMITTED'] = df['CASE_SUBMITTED'].median()
    # print(18, datetime.now().strftime("%H:%M:%S"))

    # Duration of employment
    # df['EMPLOYMENT_LENGTH'] = ((df['PERIOD_OF_EMPLOYMENT_END_DATE']
    #                             - df['PERIOD_OF_EMPLOYMENT_START_DATE'])
    #                             / np.timedelta64(1,'D'))
    # df = df.drop(columns=['PERIOD_OF_EMPLOYMENT_START_DATE',
    #                       'PERIOD_OF_EMPLOYMENT_END_DATE'])
    # df.loc[df['EMPLOYMENT_LENGTH'].isna(), 'EMPLOYMENT_LENGTH'] = df['EMPLOYMENT_LENGTH'].median()
    # print(19, datetime.now().strftime("%H:%M:%S"))

    # Number of applications submitted by company
    employer_count = pd.DataFrame(df.groupby(['EMPLOYER_NAME', 'YEAR']).size()).rename(columns={0: 'TOTAL_ANNUAL_APPLICATIONS_BY_EMPLOYER'})
    df = df.merge(employer_count,
                  left_on=['EMPLOYER_NAME', 'YEAR'],
                  right_index=True,
                  how='left',
                  validate='m:1')
    df['TOTAL_ANNUAL_APPLICATIONS_BY_EMPLOYER'] = df['TOTAL_ANNUAL_APPLICATIONS_BY_EMPLOYER'].fillna(1)
    df = df.drop(columns='EMPLOYER_NAME')
    print(20, datetime.now().strftime("%H:%M:%S"))

    # Remove applications where the company submitted 2 or fewer applications
    df = df[df["TOTAL_ANNUAL_APPLICATIONS_BY_EMPLOYER"] > 3]

    # FINISHED PREPROCESSING
    return df


def undersample(dataframe):
    df = dataframe.copy()

    rus = RandomUnderSampler(random_state=20170217)

    x, y = rus.fit_resample(df.drop(columns='CASE_STATUS'), df['CASE_STATUS'])

    x['CASE_STATUS'] = y

    return x
