import pandas as pd
def mapper(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()

    # conforming_loan_limit
    temp_df['conforming_loan_limit'] = temp_df['conforming_loan_limit'].map({
        'C': 'Conforming',
        'NC': 'Non-conforming',
        'U': 'Undetermined'})
    temp_df['conforming_loan_limit'] = (temp_df['conforming_loan_limit'].fillna('Missing')
                                        .astype(str).str.strip().str.replace(' ', '_', regex=False))


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
        3: 'Application denied'})
    temp_df['action_taken'] = (temp_df['action_taken'].fillna('Missing')
                               .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # preapproval
    temp_df['preapproval'] = temp_df['preapproval'].map({
        1: 'Preapproval requested',
        2: 'Not requested'})
    temp_df['preapproval'] = (temp_df['preapproval'].fillna('Missing')
                              .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # loan_type
    temp_df['loan_type'] = temp_df['loan_type'].map({
        1: 'Conventional',
        2: 'FHA',
        3: 'VA',
        4: 'USDA RHS/FSA'})
    temp_df['loan_type'] = (temp_df['loan_type'].fillna('Missing')
                            .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # lien_status
    temp_df['lien_status'] = temp_df['lien_status'].map({
        1: 'First lien',
        2: 'Subordinate'})
    temp_df['lien_status'] = (temp_df['lien_status'].fillna('Missing')
                              .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # reverse_mortgage
    temp_df['reverse_mortgage'] = temp_df['reverse_mortgage'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt' })
    temp_df['reverse_mortgage'] = (temp_df['reverse_mortgage'].fillna('Missing')
                                   .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # open-end_line_of_credit
    temp_df['open-end_line_of_credit'] = temp_df['open-end_line_of_credit'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt' })
    temp_df['open-end_line_of_credit'] = (temp_df['open-end_line_of_credit'].fillna('Missing')
                                          .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # business_or_commercial_purpose
    temp_df['business_or_commercial_purpose'] = temp_df['business_or_commercial_purpose'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt'})
    temp_df['business_or_commercial_purpose'] = (temp_df['business_or_commercial_purpose'].fillna('Missing')
                                                 .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # business_or_commercial_purpose
    temp_df['hoepa_status'] = temp_df['hoepa_status'].map({
        1: 'High-cost mortgage',
        2: 'Not high-cost',
        1111: 'Exempt'})
    temp_df['hoepa_status'] = (temp_df['hoepa_status'].fillna('Missing')
                               .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # negative_amortization
    temp_df['negative_amortization'] = temp_df['negative_amortization'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt'})
    temp_df['negative_amortization'] = (temp_df['negative_amortization'].fillna('Missing')
                                        .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # interest_only_payment
    temp_df['interest_only_payment'] = temp_df['interest_only_payment'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt'})
    temp_df['interest_only_payment'] = (temp_df['interest_only_payment'].fillna('Missing')
                                        .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # balloon_payment
    temp_df['balloon_payment'] = temp_df['balloon_payment'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt' })
    temp_df['balloon_payment'] = (temp_df['balloon_payment'].fillna('Missing')
                                  .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # balloon_payment
    temp_df['other_nonamortizing_features'] = temp_df['other_nonamortizing_features'].map({
        1: 'Yes',
        2: 'No',
        1111: 'Exempt'})
    temp_df['other_nonamortizing_features'] = (temp_df['other_nonamortizing_features'].fillna('Missing')
                                               .astype(str).str.strip().str.replace(' ', '_', regex=False))

    # construction_method
    temp_df['construction_method'] = temp_df['construction_method'].map({
        1: 'Site-built',
        2: 'Manufactured home'})
    temp_df['construction_method'] = (temp_df['construction_method'].fillna('Missing')
                                      .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # occupancy_type
    temp_df['occupancy_type'] = temp_df['occupancy_type'].map({
        1: 'Site-built',
        2: 'Second residence',
        3: 'Investment property'})
    temp_df['occupancy_type'] = (temp_df['occupancy_type'].fillna('Missing')
                                 .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # manufactured_home_secured_property_type
    temp_df['manufactured_home_secured_property_type'] = temp_df['manufactured_home_secured_property_type'].map({
        1: 'Home and land',
        2: 'Home not land',
        3: 'Not applicable',
        1111: 'Exempt'})
    temp_df['manufactured_home_secured_property_type'] = (
        temp_df['manufactured_home_secured_property_type'].fillna('Missing')
        .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # manufactured_home_land_property_interest
    temp_df['manufactured_home_land_property_interest'] = (
        temp_df['manufactured_home_land_property_interest'].map({
        1: 'Direct ownership',
        2: 'Indirect ownership',
        3: 'Paid leasehold',
        4: 'Unpaid leasehold',
        5: 'Not applicable',
        1111: 'Exempt'}))
    temp_df['manufactured_home_land_property_interest'] = (
        temp_df['manufactured_home_land_property_interest'].fillna('Missing')
        .astype(str).str.strip().str.replace(' ', '_', regex=False))


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
        1111: 'Exempt'})
    temp_df['applicant_credit_score_type'] = (
        temp_df['applicant_credit_score_type'].fillna('Missing')
        .astype(str).str.strip().str.replace(' ', '_', regex=False))


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
        1111: 'Exempt'})
    temp_df['co-applicant_credit_score_type'] = (
        temp_df['co-applicant_credit_score_type'].fillna('Missing')
        .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # applicant_ethnicity - 1
    temp_df['applicant_ethnicity-1'] = temp_df['applicant_ethnicity-1'].map({
        1.0: 'Hispanic or Latino',
        2.0: 'Not Hispanic or Latino',
        3.0: 'Not provided',
        4.0: 'Not applicable',
        11.0: 'Mexican',
        12.0: 'Puerto Rican',
        13.0: 'Cuban',
        14.0: 'Other Hispanic or Latino'})
    temp_df['applicant_ethnicity-1'] = (
        temp_df['applicant_ethnicity-1'].fillna('Missing')
        .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # co-applicant_ethnicity-1
    temp_df['co-applicant_ethnicity-1'] = temp_df['co-applicant_ethnicity-1'].map({
        1.0: 'Hispanic or Latino',
        2.0: 'Not Hispanic or Latino',
        3.0: 'Not provided',
        4.0: 'Not applicable',
        5.0: 'No co-applicant',
        11.0: 'Mexican',
        12.0: 'Puerto Rican',
        13.0: 'Cuban',
        14.0: 'Other Hispanic or Latino'})
    temp_df['co-applicant_ethnicity-1'] = (temp_df['co-applicant_ethnicity-1'].fillna('Missing')
                                           .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # applicant_ethnicity_observed
    temp_df['applicant_ethnicity_observed'] = temp_df['applicant_ethnicity_observed'].map({
        1: 'Collected on visual observation or surname',
        2: 'Not collected on visual observation or surname',
        3: 'Not applicable',
        1111: 'Exempt'})
    temp_df['applicant_ethnicity_observed'] = (temp_df['applicant_ethnicity_observed'].fillna('Missing')
                                               .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # co-applicant_ethnicity_observed
    temp_df['co-applicant_ethnicity_observed'] = temp_df['co-applicant_ethnicity_observed'].map({
        1: 'Collected on visual observation or surname',
        2: 'Not collected on visual observation or surname',
        3: 'Not applicable',
        4: 'No co-applicant',
        1111: 'Exempt'})
    temp_df['co-applicant_ethnicity_observed'] = (temp_df['co-applicant_ethnicity_observed'].fillna('Missing')
                                                  .astype(str).str.strip().str.replace(' ', '_', regex=False))

    # applicant_race-1
    temp_df['applicant_race-1'] = pd.to_numeric(temp_df['applicant_race-1'], errors='coerce')
    temp_df['applicant_race-1'] = temp_df['applicant_race-1'].map({
        1.0: 'American Indian or Alaska Native',
        2.0: 'Asian',
        2.1: 'Asian Indian',
        2.2: 'Chinese',
        2.3: 'Filipino',
        2.4: 'Japanese',
        2.5: 'Korean',
        2.6: 'Vietnamese',
        2.7: 'Other Asian',
        3.0: 'Black or African American',
        4.0: 'Native Hawaiian or Other Pacific Islander',
        4.1: 'Native Hawaiian',
        4.2: 'Guamanian or Chamorro',
        4.3: 'Samoan',
        4.4: 'Other Pacific Islander',
        5.0: 'White',
        6.0: 'Information not provided by applicant in mail, internet, or telephone application',
        7.0: 'Not applicable'})
    temp_df['applicant_race-1'] = (temp_df['applicant_race-1'].fillna('Missing')
                                   .astype(str).str.strip().str.replace(' ', '_', regex=False))

    # co-applicant_race-1
    temp_df['co-applicant_race-1'] = pd.to_numeric(temp_df['co-applicant_race-1'], errors='coerce')
    temp_df['co-applicant_race-1'] = temp_df['co-applicant_race-1'].map({
        1.0: 'American Indian or Alaska Native',
        2.0: 'Asian',
        2.1: 'Asian Indian',
        2.2: 'Chinese',
        2.3: 'Filipino',
        2.4: 'Japanese',
        2.5: 'Korean',
        2.6: 'Vietnamese',
        2.7: 'Other Asian',
        3.0: 'Black or African American',
        4.0: 'Native Hawaiian or Other Pacific Islander',
        4.1: 'Native Hawaiian',
        4.2: 'Guamanian or Chamorro',
        4.3: 'Samoan',
        4.4: 'Other Pacific Islander',
        5.0: 'White',
        6.0: 'Information not provided by applicant in mail, internet, or telephone application',
        7.0: 'Not applicable',
        8.0: 'No co-applicant'})
    temp_df['co-applicant_race-1'] = (temp_df['co-applicant_race-1'].fillna('Missing')
                                      .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # applicant_race_observed
    temp_df['applicant_race_observed'] = temp_df['applicant_race_observed'].map({
        1: 'Collected on visual observation or surname',
        2: 'Not collected on visual observation or surname',
        3: 'Not applicable',
        1111: 'Exempt'})
    temp_df['applicant_race_observed'] = (temp_df['applicant_race_observed'].fillna('Missing')
                                          .astype(str).str.strip().str.replace(' ', '_', regex=False))

    # co-applicant_race_observed
    temp_df['co-applicant_race_observed'] = temp_df['co-applicant_race_observed'].map({
        1: 'Collected on visual observation or surname',
        2: 'Not collected on visual observation or surname',
        3: 'Not applicable',
        4: 'No co-applicant',
        1111: 'Exempt'})
    temp_df['co-applicant_race_observed'] = (temp_df['co-applicant_race_observed'].fillna('Missing')
                                             .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # applicant_sex
    temp_df['applicant_sex'] = temp_df['applicant_sex'].map({
        1: 'Male',
        2: 'Female',
        3: 'Information not provided by applicant in mail, internet, or telephone application',
        4: 'Not applicable',
        6: 'Applicant selected both male and female'})
    temp_df['applicant_sex'] = (temp_df['applicant_sex'].fillna('Missing')
                                .astype(str).str.strip().str.replace(' ', '_', regex=False))

    # co-applicant_sex
    temp_df['co-applicant_sex'] = temp_df['co-applicant_sex'].map({
        1: 'Male',
        2: 'Female',
        3: 'Information not provided by applicant in mail, internet, or telephone application',
        4: 'Not applicable',
        5: 'No co-applicant',
        6: 'Co-applicant selected both male and female'})
    temp_df['co-applicant_sex'] = (temp_df['co-applicant_sex'].fillna('Missing')
                                   .astype(str).str.strip().str.replace(' ', '_', regex=False))

    # applicant_sex_observed
    temp_df['applicant_sex_observed'] = temp_df['applicant_sex_observed'].map({
        1: 'Collected on visual observation or surname',
        2: 'Not collected on visual observation or surname',
        3: 'Not applicable',
        1111: 'Exempt'})
    temp_df['applicant_sex_observed'] = (temp_df['applicant_sex_observed'].fillna('Missing')
                                         .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # co-applicant_sex_observed
    temp_df['co-applicant_sex_observed'] = temp_df['co-applicant_sex_observed'].map({
        1: 'Collected on visual observation or surname',
        2: 'Not collected on visual observation or surname',
        3: 'Not applicable',
        4: 'No co-applicant',
        1111: 'Exempt'})
    temp_df['co-applicant_sex_observed'] = (temp_df['co-applicant_sex_observed'].fillna('Missing')
                                            .astype( str).str.strip().str.replace(' ', '_', regex=False))

    age_map = {
        '<25': 1,
        '25-34': 2,
        '35-44': 3,
        '45-54': 4,
        '55-64': 5,
        '65-74': 6,
        '>74': 7
    }

    # Applicant age
    temp_df['applicant_age_ord'] = (temp_df['applicant_age'].map(age_map).astype('Int8'))
    temp_df['applicant_age_not_applicable'] = (temp_df['applicant_age'].eq('8888').astype('int8'))
    temp_df['applicant_age_ord'] = (temp_df['applicant_age_ord'].fillna(0).astype('int8'))
    temp_df = temp_df.drop(columns=['applicant_age'])

    # Co-applicant age
    temp_df['coapplicant_age_ord'] = (temp_df['co-applicant_age'].map(age_map).astype('Int8'))
    temp_df['coapplicant_age_not_applicable'] = (temp_df['co-applicant_age'].eq('8888').astype('int8'))
    temp_df['no_coapplicant'] = (temp_df['co-applicant_age'].eq('9999').astype('int8'))
    temp_df['coapplicant_age_ord'] = (temp_df['coapplicant_age_ord'].fillna(0).astype('int8'))

    temp_df = temp_df.drop(columns=['co-applicant_age'])



    # applicant_age_above_62
    temp_df['applicant_age_above_62'] = temp_df['applicant_age_above_62'].map({
        'Yes': 'Yes',
        'No': 'No',
    })
    temp_df['applicant_age_above_62'] = (temp_df['applicant_age_above_62'].fillna('Exempt')
                                         .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # submission_of_application
    temp_df['submission_of_application'] = temp_df['submission_of_application'].map({
        1: 'Submitted directly to institution',
        2: 'Not submitted directly to institution',
        1111: 'Exempt'})
    temp_df['submission_of_application'] = (temp_df['submission_of_application'].fillna('Missing')
                                            .astype(str).str.strip().str.replace(' ', '_', regex=False))


    # initially_payable_to_institution
    temp_df['initially_payable_to_institution'] = temp_df['initially_payable_to_institution'].map({
        1: 'Initially payable to your institution',
        2: 'Not initially payable to your institution',
        3: 'Not applicable',
        1111: 'Exempt'})
    temp_df['initially_payable_to_institution'] = (temp_df['initially_payable_to_institution'].fillna('Missing')
                                                   .astype(str).str.strip().str.replace(' ', '_', regex=False))

    # aus-1
    temp_df['aus-1'] = temp_df['aus-1'].map({
        1: 'Desktop Underwriter (DU)',
        2: 'Loan Product Advisor (LPA)',
        3: 'GUS',
        4: 'Other',
        5: 'Not applicable',
        6: 'Internal Proprietary AUS',
        1111: 'Exempt'})
    temp_df['aus-1'] = (temp_df['aus-1'].fillna('Missing')
                        .astype(str).str.strip().str.replace(' ', '_', regex=False))

    return temp_df
