import pandas as pd
def extract_date_components(df: pd.DataFrame,
                          date_field: str,
                          include_time: bool = False,
                          date_format: str = '%Y-%m-%dT%H:%M:%S') -> pd.DataFrame:
    """
    Extract year, month, and day components from a datetime column in a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        date_field (str): Name of the column containing dates
        include_time (bool): Whether to include hour, minute, second extraction
        date_format (str): Format string for parsing dates (default: '%Y-%m-%dT%H:%M:%S')

    Returns:
        pd.DataFrame: Modified DataFrame with added date/time component columns
    """
    # Ensure the date field exists in the DataFrame
    if date_field not in df.columns:
        raise ValueError(f"The column '{date_field}' does not exist in the DataFrame")

    # Convert the date field to datetime using the specified format
    df[date_field] = pd.to_datetime(df[date_field], format=date_format)

    # Extract basic date components
    df[f"{date_field}_year"] = df[date_field].dt.year
    df[f"{date_field}_month"] = df[date_field].dt.month
    df[f"{date_field}_day"] = df[date_field].dt.day

    # If include_time is True, extract time components
    if include_time:
        df[f"{date_field}_hour"] = df[date_field].dt.hour
        df[f"{date_field}_minute"] = df[date_field].dt.minute
        df[f"{date_field}_second"] = df[date_field].dt.second

    return df