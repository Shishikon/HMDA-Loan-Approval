import pandas as pd


def auditor(df: pd.DataFrame) -> pd.DataFrame:
    audit_data = []

    for col in df.columns:
        dtype = df[col].dtypes
        missing_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        audit_data.append({
            'column': col,
            'dtype': str(dtype),
            'missing_count': missing_count,
            'unique_val': unique_count
        })
    audit_df = pd.DataFrame(audit_data)
    return audit_df


def summarizer(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()

    summary_data = {
        'Data Type': temp_df.dtypes,
        'Non-Null Count': temp_df.notnull().sum(),
        'Missing Count': temp_df.isnull().sum(),
        'Unique Values': temp_df.nunique(),
        'Sample Values': [
            temp_df[col].dropna().unique()[:1].tolist()
            for col in temp_df.columns]}

    summary_df = pd.DataFrame(summary_data)

    return summary_df