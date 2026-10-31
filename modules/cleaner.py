"""
                                                                             0
derived_msa-md                                                           42540
county_code                                                            42069.0
census_tract                                                     42069100400.0
conforming_loan_limit                                                        C
derived_loan_product_type                                       FHA:First Lien
derived_dwelling_category                 Single Family (1-4 Units):Site-Built
derived_ethnicity                                       Not Hispanic or Latino
derived_race                                                             Asian
derived_sex                                                              Joint
action_taken                                                                 1
preapproval                                                                  2
loan_type                                                                    2
lien_status                                                                  1
reverse_mortgage                                                             2
open-end_line_of_credit                                                      2
business_or_commercial_purpose                                               2
loan_amount                                                            95000.0
loan_to_value_ratio                                                   62.09400
interest_rate                                                            3.250
rate_spread                                                            Missing
hoepa_status                                                                 2
total_loan_costs                                                       3456.19
origination_charges                                                    1800.00
loan_term                                                                  304
negative_amortization                                                        2
interest_only_payment                                                        2
balloon_payment                                                              2
other_nonamortizing_features                                                 2
property_value                                                          155000
construction_method                                                          1
occupancy_type                                                               1
manufactured_home_secured_property_type                                      3
manufactured_home_land_property_interest                                     5
total_units                                                                  1
income                                                                    59.0
debt_to_income_ratio                                                      <20%
applicant_credit_score_type                                                  1
co-applicant_credit_score_type                                               9
applicant_ethnicity-1                                                      2.0
co-applicant_ethnicity-1                                                   2.0
applicant_ethnicity_observed                                                 2
co-applicant_ethnicity_observed                                              2
applicant_race-1                                                           2.0
co-applicant_race-1                                                        2.0
applicant_race_observed                                                      2
co-applicant_race_observed                                                   2
applicant_sex                                                                1
co-applicant_sex                                                             2
applicant_sex_observed                                                       2
co-applicant_sex_observed                                                    2
applicant_age                                                            55-64
co-applicant_age                                                         55-64
applicant_age_above_62                                                      No
submission_of_application                                                    1
initially_payable_to_institution                                             1
aus-1                                                                        6
tract_population                                                          2039
tract_minority_population_percent                                        45.12
ffiec_msa_md_median_family_income                                        83000
tract_to_msa_income_percentage                                             108
tract_owner_occupied_units                                                 443
tract_one_to_four_family_homes                                            1062
tract_median_age_of_housing_units                                            0



"""

import time

import pandas as pd
import numpy as np
from modules.audit import auditor


pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

raw = pd.read_csv('../actions_taken_1-2-3_loan_purposes_1_state_PA.csv', low_memory=False)

copy = raw.copy()






# Unique value
def one_value(df: pd.DataFrame) -> pd.DataFrame:
    uniq_1 = df.columns[df.nunique() == 1]
    return df.drop(columns=uniq_1)
copy = one_value(copy)



# High cardinality
def cardinals(df: pd.DataFrame) -> pd.DataFrame:
    cardinal = "lei"
    return df.drop(columns=cardinal)
copy = cardinals(copy)



# Data leakage
def leaker(df: pd.DataFrame) -> pd.DataFrame:
    leakers = ["purchaser_type", "denial_reason-1", "denial_reason-2",
               "denial_reason-3", "denial_reason-4"]
    return df.drop(columns=leakers, errors="ignore")
copy = leaker(copy)


# High missingness
def high_missing(df: pd.DataFrame) -> pd.DataFrame:
    missers = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_rate = (missing_count / df.shape[0]) * 100
        if missing_rate >= 50:
            missers.append(col)
    return df.drop(columns=missers, errors="ignore")

copy = high_missing(copy)



# Mapping
from mapper import mapper

# copy = mapper(copy) -> after mapper, uncomment
print(mapper(copy))















# DataType Transformation
def transformer(df: pd.DataFrame) -> pd.DataFrame:
    change = ['loan_to_value_ratio', 'interest_rate', 'rate_spread',
              'total_loan_costs', 'origination_charges', 'loan_term',
              'property_value', 'debt_to_income_ratio']
    pass


# print(copy['derived_loan_product_type'].value_counts())






















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