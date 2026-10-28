"""
HMDA 2024 public LAR code dictionary (offline copy).
Source: CFPB/FFIEC, "Public HMDA - LAR Data Fields"
https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields
Cite the page and your access date in the report.
"""

YES_NO = {1: "Yes", 2: "No", 1111: "Exempt"}          # flag columns: 1 = feature present

ETHNICITY = {1: "Hispanic or Latino", 11: "Mexican", 12: "Puerto Rican", 13: "Cuban",
             14: "Other Hispanic or Latino", 2: "Not Hispanic or Latino",
             3: "Not provided", 4: "Not applicable", 5: "No co-applicant"}

RACE = {1: "American Indian or Alaska Native", 2: "Asian", 21: "Asian Indian", 22: "Chinese",
        23: "Filipino", 24: "Japanese", 25: "Korean", 26: "Vietnamese", 27: "Other Asian",
        3: "Black or African American", 4: "Native Hawaiian or Other Pacific Islander",
        41: "Native Hawaiian", 42: "Guamanian or Chamorro", 43: "Samoan",
        44: "Other Pacific Islander", 5: "White", 6: "Not provided",
        7: "Not applicable", 8: "No co-applicant"}

SEX = {1: "Male", 2: "Female", 3: "Not provided", 4: "Not applicable",
       5: "No co-applicant", 6: "Both male and female"}

OBSERVED = {1: "Observed/surname", 2: "Not observed", 3: "Not applicable", 4: "No co-applicant"}

AUS = {1: "Desktop Underwriter (DU)", 2: "Loan Prospector (LP)", 3: "TOTAL Scorecard",
       4: "GUS", 5: "Other", 6: "Not applicable", 7: "Internal proprietary", 1111: "Exempt"}

CREDIT_SCORE_TYPE = {1: "Equifax Beacon 5.0", 2: "Experian Fair Isaac", 3: "FICO Classic 04",
                     4: "FICO Classic 98", 5: "VantageScore 2.0", 6: "VantageScore 3.0",
                     7: "More than one model", 8: "Other model", 9: "Not applicable",
                     10: "No co-applicant", 11: "FICO Score 9", 12: "FICO Score 8",
                     13: "FICO Score 10", 14: "FICO Score 10T", 15: "VantageScore 4.0",
                     1111: "Exempt"}

DENIAL_REASON = {1: "Debt-to-income ratio", 2: "Employment history", 3: "Credit history",
                 4: "Collateral", 5: "Insufficient cash", 6: "Unverifiable information",
                 7: "Credit application incomplete", 8: "Mortgage insurance denied",
                 9: "Other", 10: "Not applicable", 1111: "Exempt"}

CODES = {
    "action_taken": {1: "Loan originated", 2: "Approved but not accepted", 3: "Application denied",
                     4: "Withdrawn by applicant", 5: "File closed for incompleteness",
                     6: "Purchased loan", 7: "Preapproval denied",
                     8: "Preapproval approved but not accepted"},
    "purchaser_type": {0: "Not applicable", 1: "Fannie Mae", 2: "Ginnie Mae", 3: "Freddie Mac",
                       4: "Farmer Mac", 5: "Private securitizer",
                       6: "Commercial/savings bank", 71: "Credit union/mortgage/finance co.",
                       72: "Life insurance company", 8: "Affiliate institution", 9: "Other"},
    "preapproval": {1: "Preapproval requested", 2: "Not requested"},
    "loan_type": {1: "Conventional", 2: "FHA", 3: "VA", 4: "USDA RHS/FSA"},
    "loan_purpose": {1: "Home purchase", 2: "Home improvement", 31: "Refinancing",
                     32: "Cash-out refinancing", 4: "Other", 5: "Not applicable"},
    "lien_status": {1: "First lien", 2: "Subordinate lien"},
    "hoepa_status": {1: "High-cost mortgage", 2: "Not high-cost", 3: "Not applicable"},
    "construction_method": {1: "Site-built", 2: "Manufactured home"},
    "occupancy_type": {1: "Principal residence", 2: "Second residence", 3: "Investment property"},
    "manufactured_home_secured_property_type": {1: "Home and land", 2: "Home, not land",
                                                3: "Not applicable", 1111: "Exempt"},
    "manufactured_home_land_property_interest": {1: "Direct ownership", 2: "Indirect ownership",
                                                 3: "Paid leasehold", 4: "Unpaid leasehold",
                                                 5: "Not applicable", 1111: "Exempt"},
    "submission_of_application": {1: "Direct to institution", 2: "Not direct",
                                  3: "Not applicable", 1111: "Exempt"},
    "initially_payable_to_institution": {1: "Payable to institution", 2: "Not payable",
                                         3: "Not applicable", 1111: "Exempt"},
    "applicant_credit_score_type": CREDIT_SCORE_TYPE,
    "co-applicant_credit_score_type": CREDIT_SCORE_TYPE,
    "applicant_ethnicity_observed": OBSERVED, "co-applicant_ethnicity_observed": OBSERVED,
    "applicant_race_observed": OBSERVED, "co-applicant_race_observed": OBSERVED,
    "applicant_sex": SEX, "co-applicant_sex": SEX,
    "applicant_sex_observed": OBSERVED, "co-applicant_sex_observed": OBSERVED,
}
for c in ["reverse_mortgage", "open-end_line_of_credit", "business_or_commercial_purpose",
          "negative_amortization", "interest_only_payment", "balloon_payment",
          "other_nonamortizing_features"]:
    CODES[c] = YES_NO
for i in range(1, 6):
    CODES[f"applicant_ethnicity-{i}"] = ETHNICITY
    CODES[f"co-applicant_ethnicity-{i}"] = ETHNICITY
    CODES[f"applicant_race-{i}"] = RACE
    CODES[f"co-applicant_race-{i}"] = RACE
    CODES[f"aus-{i}"] = AUS
for i in range(1, 5):
    CODES[f"denial_reason-{i}"] = DENIAL_REASON


def decode(df, columns=None, suffix="_label"):
    """Add readable '<col>_label' columns. Originals are kept. Unmapped codes become NaN."""
    out = df.copy()
    for c in (columns or [c for c in CODES if c in df.columns]):
        out[c + suffix] = out[c].map(CODES[c])
    return out


def check_coverage(df):
    """Return {column: codes found in your data but missing from the dictionary}."""
    gaps = {}
    for c, m in CODES.items():
        if c in df.columns:
            missing = sorted(set(df[c].dropna().unique()) - set(m))
            if missing:
                gaps[c] = missing
    return gaps