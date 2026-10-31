import pandas as pd
import numpy as np




def mapper(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    to_map = ['applicant_ethnicity-1',
              'co-applicant_ethnicity-1', 'applicant_ethnicity_observed',
              'co-applicant_ethnicity_observed', 'applicant_race-1',
              'co-applicant_race-1', 'applicant_race_observed',
              'co-applicant_race_observed', 'applicant_sex',
              'co-applicant_sex', 'applicant_sex_observed',
              'co-applicant_sex_observed', 'applicant_age_above_62',
              'submission_of_application', 'initially_payable_to_institution',
              'aus-1']

    # conforming_loan_limit
    temp_df['conforming_loan_limit'] = temp_df['conforming_loan_limit'].map({
        'C': 'Conforming',
        'NC': 'Non-conforming',
        'U': 'Exempt'})
    temp_df['conforming_loan_limit'] = temp_df['conforming_loan_limit'].fillna('Missing')

    # derived_dwelling_category
    temp_df['derived_dwelling_category'] = temp_df['derived_dwelling_category'].map({
        "Single Family (1-4 Units):Site-Built": 1,
        "Single Family (1-4 Units):Manufactured": 2,
        "Multifamily:Site-Built": 3,
        "Multifamily:Manufactured": 4})
    temp_df['derived_dwelling_category'] = temp_df['derived_dwelling_category'].fillna(0).astype(int)

    # action_taken
    temp_df['action_taken'] = temp_df['action_taken'].map({
        1: 'Loan originated',
        2: 'Approved but not accepted',
        3: 'Application denied',
    })
    temp_df['action_taken'] = temp_df['action_taken'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # preapproval
    temp_df['preapproval'] = temp_df['preapproval'].map({
        1: 'Preapproval requested',
        2: 'Not requested'
    })
    temp_df['preapproval'] = temp_df['preapproval'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # loan_type
    temp_df['loan_type'] = temp_df['loan_type'].map({
        1: 'Conventional',
        2: 'FHA',
        3: 'VA',
        4: 'USDA RHS/FSA'
    })
    temp_df['loan_type'] = temp_df['loan_type'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # lien_status
    temp_df['lien_status'] = temp_df['lien_status'].map({
        1: 'First lien',
        2: 'Subordinate'
    })
    temp_df['lien_status'] = temp_df['lien_status'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # reverse_mortgage
    temp_df['reverse_mortgage'] = temp_df['reverse_mortgage'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt'
    })
    temp_df['reverse_mortgage'] = temp_df['reverse_mortgage'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # open-end_line_of_credit
    temp_df['open-end_line_of_credit'] = temp_df['open-end_line_of_credit'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt'
    })
    temp_df['open-end_line_of_credit'] = temp_df['open-end_line_of_credit'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # business_or_commercial_purpose
    temp_df['business_or_commercial_purpose'] = temp_df['business_or_commercial_purpose'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt'
    })
    temp_df['business_or_commercial_purpose'] = temp_df['business_or_commercial_purpose'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # business_or_commercial_purpose
    temp_df['hoepa_status'] = temp_df['hoepa_status'].map({
        1: 'High-cost mortgage',
        2: 'Not high-cost',
        1111: 'Exempt'
    })
    temp_df['hoepa_status'] = temp_df['hoepa_status'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # negative_amortization
    temp_df['negative_amortization'] = temp_df['negative_amortization'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt'
    })
    temp_df['negative_amortization'] = temp_df['negative_amortization'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # interest_only_payment
    temp_df['interest_only_payment'] = temp_df['interest_only_payment'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt'
    })
    temp_df['interest_only_payment'] = temp_df['interest_only_payment'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)



    # balloon_payment
    temp_df['balloon_payment'] = temp_df['balloon_payment'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt'
    })
    temp_df['balloon_payment'] = temp_df['balloon_payment'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # balloon_payment
    temp_df['other_nonamortizing_features'] = temp_df['other_nonamortizing_features'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt'
    })
    temp_df['other_nonamortizing_features'] = temp_df['other_nonamortizing_features'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)

    # construction_method
    temp_df['construction_method'] = temp_df['construction_method'].map({
        1: 'Site-built',
        2: 'Manufactured home'
    })
    temp_df['construction_method'] = temp_df['construction_method'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # occupancy_type
    temp_df['occupancy_type'] = temp_df['occupancy_type'].map({
        1: 'Site-built',
        2: 'Second residence',
        3: 'Investment property'
    })
    temp_df['occupancy_type'] = temp_df['occupancy_type'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # manufactured_home_secured_property_type
    temp_df['manufactured_home_secured_property_type'] = temp_df['manufactured_home_secured_property_type'].map({
        1: 'Home and land',
        2: 'Home not land',
        3: 'Not applicable',
        1111: 'Exempt'
    })
    temp_df['manufactured_home_secured_property_type'] = temp_df['manufactured_home_secured_property_type'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # manufactured_home_land_property_interest
    temp_df['manufactured_home_land_property_interest'] = temp_df['manufactured_home_land_property_interest'].map({
        1: 'Direct ownership',
        2: 'Indirect ownership',
        3: 'Paid leasehold',
        4: 'Unpaid leasehold',
        5: 'Not applicable',
        1111: 'Exempt'
    })
    temp_df['manufactured_home_land_property_interest'] = temp_df['manufactured_home_land_property_interest'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)


    # applicant_credit_score_type
    temp_df['applicant_credit_score_type'] = temp_df['applicant_credit_score_type'].map({
        1: 'Equifax Beacon 5.0',
        2: 'Experian Fair Isaac',
        3: 'FICO Classic 04',
        4: 'FICO Classic 98',
        5: 'VantageScore 2.0',
        6: 'VantageScore 3.0',
        7: 'More than one model',
        8: 'Other model',
        9: 'Not applicable',
        11: 'FICO Score 9',
        12: 'FICO Score 8',
        13: 'FICO Score 10',
        14: 'FICO Score 10T',
        15: 'VantageScore 4.0',
        1111: 'Exempt',
    })
    temp_df['applicant_credit_score_type'] = temp_df['applicant_credit_score_type'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)

    # co-applicant_credit_score_type
    temp_df['co-applicant_credit_score_type'] = temp_df['co-applicant_credit_score_type'].map({
        1: 'Equifax Beacon 5.0',
        2: 'Experian Fair Isaac',
        3: 'FICO Classic 04',
        4: 'FICO Classic 98',
        5: 'VantageScore 2.0',
        6: 'VantageScore 3.0',
        7: 'More than one model',
        8: 'Other model',
        9: 'Not applicable',
        10: 'No co-applicant',
        11: 'FICO Score 9',
        12: 'FICO Score 8',
        13: 'FICO Score 10',
        14: 'FICO Score 10T',
        15: 'VantageScore 4.0',
        1111: 'Exempt',
    })
    temp_df['co-applicant_credit_score_type'] = temp_df['co-applicant_credit_score_type'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)

    temp_df['applicant_ethnicity-1'] = temp_df['applicant_ethnicity-1'].map({
        1.0: 'Hispanic or Latino',
        2.0: 'Not Hispanic or Latino',
        3.0: 'Not provided',
        4.0: 'Not applicable',
        11.0: 'Mexican',
        12.0: 'Puerto Rican',
        13.0: 'Cuban',
        14.0: 'Other Hispanic or Latino',

    })

    temp_df['applicant_ethnicity-1'] = temp_df['applicant_ethnicity-1'].fillna('Missing').astype(str).str.strip().str.replace(' ', '_', regex=False)
    return temp_df['applicant_ethnicity-1'].value_counts()



