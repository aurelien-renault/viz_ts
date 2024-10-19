import pandas as pd

def check_df_format(df, x_param=True):

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Input shoule be a pandas Dataframe, received {type(df)}")

    # check column names 
    cols_to_check = ["dataset", "competitor", "metric"]
    if x_param:
        cols_to_check += ["x_parameter"]
    for col in df.columns:
        if col not in cols_to_check:
            raise ValueError(f"Column '{col}' not found in provided dataframe")
        
    # check same number of experiments for each competitor
    n_dataset = df.groupby("competitor")["dataset"].apply(len)
    if len(n_dataset.unique()) != 1:
        raise ValueError("Not the same number of results for each competitor !")
