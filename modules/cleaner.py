import pandas as pd

raw = pd.read_csv('../actions_taken_1-2-3_loan_purposes_1_state_PA.csv', low_memory=False)

copy = raw.copy()

# Unique value
def one_value(df: pd.DataFrame) -> pd.DataFrame:
    uniq_1 = df.columns[df.nunique() == 1]
    return df.drop(columns=uniq_1)




# High cardinality
def cardinals(df: pd.DataFrame) -> pd.DataFrame:
    cardinal = "lei"
    return df.drop(columns=cardinal)




# Data leakage
def leaker(df: pd.DataFrame) -> pd.DataFrame:
    leakers = ["purchaser_type", "denial_reason-1", "denial_reason-2",
               "denial_reason-3", "denial_reason-4"]
    return df.drop(columns=leakers, errors="ignore")



# High missingness
def high_missing(df: pd.DataFrame) -> pd.DataFrame:
    missers = []
    for colum in df.columns:
        missing_count = df[colum].isna().sum()
        missing_rate = (missing_count / df.shape[0]) * 100
        if missing_rate >= 50:
            missers.append(colum)
    return df.drop(columns=missers, errors="ignore")





# DataType Transformation
def transformer(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    from_str_to_float = ['loan_to_value_ratio', 'interest_rate', 'rate_spread',
                         'total_loan_costs', 'origination_charges', 'loan_term',
                         'property_value', 'debt_to_income_ratio']

    for string in from_str_to_float:
        if string in temp_df.columns:
            temp_df[string] = pd.to_numeric(temp_df[string], errors='coerce')


    from_str_to_int = ['total_units']

    return temp_df





def filler(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    miss = ['county_code','census_tract',
            'loan_to_value_ratio','interest_rate',
            'rate_spread','total_loan_costs',
            'origination_charges','loan_term',
            'property_value','income',
            'debt_to_income_ratio']


    median = ['county_code', 'census_tract', 'interest_rate', 'income']
    mode = ['debt_to_income_ratio']



    return temp_df





# filler(copy)





























"""
# Handle Missing Values
def imputer(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    columns = ["county_code", "census_tract", "conforming_loan_limit",
           "loan_to_value_ratio", "interest_rate", "rate_spread",
           "total_loan_costs", "origination_charges", "loan_term",
           "property_value", 'income', "debt_to_income_ratio",
           "applicant_ethnicity-1", "co-applicant_ethnicity-1",
           "applicant_race-1", "co-applicant_race-1",
           "applicant_age_above_62"]
    numeric_cols = temp_df[columns].select_dtypes(include=[np.number]).columns
    categorical_cols = temp_df[columns].select_dtypes(exclude=[np.number]).columns
    for col in numeric_cols:
        median_val = temp_df[col].median()
        temp_df[col] = temp_df[col].fillna(median_val)
    for col in categorical_cols:
        temp_df[col] = temp_df[col].fillna('Missing')
    return temp_df
copy = imputer(copy)


"""