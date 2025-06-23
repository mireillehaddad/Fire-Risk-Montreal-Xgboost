def safe_drop_columns(df,columns_to_drop):
    """ 
        Safely drops columns from df (in place). I.e if column does not exist in df it is just ignored
    """
    cols = [col for col in columns_to_drop if col in df.columns]
    df.drop(cols,axis=1,inplace=True)