"""
                                      column    dtype  missing_count  missing_rate  unique_val
0                              activity_year    int64              0          0.00           1
1                                        lei      str              0          0.00         934
2                             derived_msa-md    int64              0          0.00          18
3                                 state_code      str              0          0.00           1
4                                county_code  float64            868          0.62          67
5                               census_tract  float64           1404          1.01        3376
6                      conforming_loan_limit      str            830          0.60           3
7                  derived_loan_product_type      str              0          0.00           7
8                  derived_dwelling_category      str              0          0.00           4
9                          derived_ethnicity      str              0          0.00           5
10                              derived_race      str              0          0.00           9
11                               derived_sex      str              0          0.00           4
12                              action_taken    int64              0          0.00           3
13                            purchaser_type    int64              0          0.00          11
14                               preapproval    int64              0          0.00           2
15                                 loan_type    int64              0          0.00           4
16                              loan_purpose    int64              0          0.00           1
17                               lien_status    int64              0          0.00           2
18                          reverse_mortgage    int64              0          0.00           3
19                   open-end_line_of_credit    int64              0          0.00           3
20            business_or_commercial_purpose    int64              0          0.00           3
21                               loan_amount  float64              0          0.00         352
22                       loan_to_value_ratio      str           8355          6.00       23919
23                             interest_rate      str          15347         11.02        1934
24                               rate_spread      str          27198         19.52       11043
25                              hoepa_status    int64              0          0.00           3
26                          total_loan_costs      str          35067         25.17       91078
27                     total_points_and_fees      str         133302         95.69         759
28                       origination_charges      str          33371         23.96       51749
29                           discount_points      str          81076         58.20       39920
30                            lender_credits      str         100758         72.33       18871
31                                 loan_term      str            425          0.31         240
32                   prepayment_penalty_term      str         133925         96.14          10
33                         intro_rate_period      str         122594         88.01          66
34                     negative_amortization    int64              0          0.00           3
35                     interest_only_payment    int64              0          0.00           3
36                           balloon_payment    int64              0          0.00           3
37              other_nonamortizing_features    int64              0          0.00           3
38                            property_value      str           1975          1.42         427
39                       construction_method    int64              0          0.00           2
40                            occupancy_type    int64              0          0.00           3
41   manufactured_home_secured_property_type    int64              0          0.00           4
42  manufactured_home_land_property_interest    int64              0          0.00           6
43                               total_units      str              0          0.00           9
44              multifamily_affordable_units      str         133958         96.16          15
45                                    income  float64           8128          5.83        1284
46                      debt_to_income_ratio      str           8520          6.12          20
47               applicant_credit_score_type    int64              0          0.00          15
48            co-applicant_credit_score_type    int64              0          0.00          16
49                     applicant_ethnicity-1  float64             38          0.03           8
50                     applicant_ethnicity-2  float64         132888         95.40           6
51                     applicant_ethnicity-3  float64         139163         99.90           5
52                     applicant_ethnicity-4  float64         139300        100.00           1
53                     applicant_ethnicity-5  float64         139298        100.00           1
54                  co-applicant_ethnicity-1  float64              7          0.01           9
55                  co-applicant_ethnicity-2  float64         137504         98.71           6
56                  co-applicant_ethnicity-3  float64         139254         99.97           3
57                  co-applicant_ethnicity-4  float64         139299        100.00           1
58                  co-applicant_ethnicity-5  float64         139296        100.00           1
59              applicant_ethnicity_observed    int64              0          0.00           3
60           co-applicant_ethnicity_observed    int64              0          0.00           4
61                          applicant_race-1  float64             27          0.02          17
62                          applicant_race-2  float64         131130         94.13          16
63                          applicant_race-3  float64         138811         99.65          14
64                          applicant_race-4  float64         139257         99.97          12
65                          applicant_race-5  float64         139289         99.99           7
66                       co-applicant_race-1  float64             11          0.01          18
67                       co-applicant_race-2  float64         136090         97.69          15
68                       co-applicant_race-3  float64         139137         99.88          13
69                       co-applicant_race-4  float64         139288         99.99           9
70                       co-applicant_race-5  float64         139296        100.00           5
71                   applicant_race_observed    int64              0          0.00           3
72                co-applicant_race_observed    int64              0          0.00           4
73                             applicant_sex    int64              0          0.00           5
74                          co-applicant_sex    int64              0          0.00           6
75                    applicant_sex_observed    int64              0          0.00           3
76                 co-applicant_sex_observed    int64              0          0.00           4
77                             applicant_age      str              0          0.00           8
78                          co-applicant_age      str              0          0.00           9
79                    applicant_age_above_62      str           6195          4.45           2
80                 co-applicant_age_above_62      str          82797         59.44           2
81                 submission_of_application    int64              0          0.00           3
82          initially_payable_to_institution    int64              0          0.00           4
83                                     aus-1    int64              0          0.00           8
84                                     aus-2  float64         126455         90.78           6
85                                     aus-3  float64         135537         97.30           6
86                                     aus-4  float64         136943         98.31           3
87                                     aus-5  float64         137143         98.45           3
88                           denial_reason-1    int64              0          0.00          11
89                           denial_reason-2  float64         135257         97.10           9
90                           denial_reason-3  float64         138353         99.32           9
91                           denial_reason-4  float64         139065         99.83           9
92                          tract_population    int64              0          0.00        2496
93         tract_minority_population_percent  float64              0          0.00        2377
94         ffiec_msa_md_median_family_income    int64              0          0.00          19
95            tract_to_msa_income_percentage    int64              0          0.00         241
96                tract_owner_occupied_units    int64              0          0.00        1610
97            tract_one_to_four_family_homes    int64              0          0.00        1752
98         tract_median_age_of_housing_units    int64              0          0.00          69



"""""


import pandas as pd
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
cardinals = ["lei",]



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


# Handle Missing Values
def median_imputer(df: pd.DataFrame) -> pd.DataFrame:
    lll = ["county_code", "census_tract", "conforming_loan_limit",
           "loan_to_value_ratio", "interest_rate", "rate_spread",
           "total_loan_costs", "origination_charges", "loan_term",
           "property_value", 'income', "debt_to_income_ratio",
           "applicant_ethnicity-1", "co-applicant_ethnicity-1",
           "applicant_race-1", "co-applicant_race-1",
           "applicant_age_above_62"]
    # print(df[lll].head(3).to_string())
    return df
median_imputer(copy)
