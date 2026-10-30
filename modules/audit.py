import pandas as pd


def auditor(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    audit_data = []

    for col in df.columns:
        dtype = df[col].dtypes
        missing_count = df[col].isna().sum()
        missing_rate = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        unique_count = df[col].nunique()
        audit_data.append({
            'column': col,
            'dtype': str(dtype),
            'missing_count': missing_count,
            'missing_rate': round(missing_rate, 2),
            'unique_val': unique_count,
        })
    audit_df = pd.DataFrame(audit_data)
    return audit_df