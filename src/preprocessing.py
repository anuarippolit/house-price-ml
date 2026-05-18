import numpy as np 
import pandas as pd 

def remove_outliers(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df_cleaned = df.copy()

    if is_train:
        df_cleaned = df_cleaned[df_cleaned['GrLivArea'] < 4000]
    
    return df_cleaned

def log_transform_target(target: pd.Series) -> pd.Series:
    return np.log1p(target)