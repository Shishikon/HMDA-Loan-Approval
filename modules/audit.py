import pandas as pd


def auditor(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    audit_data = []

    for col in df.columns:
        dtype = df[col].dtypes
        unique_count = df[col].nunique()
        audit_data.append({
            'column': col,
            'dtype': str(dtype),
            'unique_val': unique_count,
        })
    audit_df = pd.DataFrame(audit_data)
    return audit_df