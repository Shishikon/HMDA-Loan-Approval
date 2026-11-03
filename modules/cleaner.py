import pandas as pd

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
    for string in from_str_to_int:
        if string in temp_df.columns:
            temp_df[string] = pd.to_numeric(temp_df[string], errors='coerce')
    return temp_df



def filler(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    mode = ['county_code','census_tract',
            'loan_to_value_ratio','interest_rate',
            'rate_spread','total_loan_costs',
            'origination_charges','loan_term',
            'property_value','income',
            'debt_to_income_ratio']

    for col in mode:
        if col in temp_df.columns:
            if temp_df[col].isna().any():
                temp_df[col] = temp_df[col].fillna(temp_df[col].mode()[0])
    return temp_df
