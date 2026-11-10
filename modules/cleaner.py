import pandas as pd


def one_value(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that have only a single unique value — they carry no information."""
    uniq_1 = df.columns[df.nunique() == 1]
    return df.drop(columns=uniq_1)


def cardinals(df: pd.DataFrame) -> pd.DataFrame:
    """Drop extremely high-cardinality identifiers that cannot be generalised.
    'lei' is a unique legal entity identifier per institution — effectively an ID column."""
    cardinal = ["lei"]
    return df.drop(columns=cardinal)


def leaker(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that would cause data leakage.
    purchaser_type is only known after a loan is sold (post-decision).
    denial_reason-* directly encode the target outcome and would leak the label."""
    leakers = ["purchaser_type", "denial_reason-1", "denial_reason-2",
               "denial_reason-3", "denial_reason-4"]
    return df.drop(columns=leakers, errors="ignore")


def high_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns where more than 50% of values are missing.
    At that level of missingness, imputation would be manufacturing data
    rather than recovering it, introducing more noise than signal."""
    missers = [col for col in df.columns
               if (df[col].isna().sum() / df.shape[0]) * 100 >= 50]
    return df.drop(columns=missers, errors="ignore")


def transformer(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns stored as strings to their correct numeric types.
    These arrive as strings in the raw HMDA file due to exempt/NA codes
    mixed into otherwise numeric fields."""
    temp_df = df.copy()

    numeric_cols = [
        'loan_to_value_ratio', 'interest_rate', 'rate_spread',
        'total_loan_costs', 'origination_charges', 'loan_term',
        'property_value', 'debt_to_income_ratio', 'total_units'
    ]

    for col in numeric_cols:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

    return temp_df