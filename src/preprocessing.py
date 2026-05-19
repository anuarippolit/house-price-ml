import numpy as np 
import pandas as pd 

def remove_outliers(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df_cleaned = df.copy()

    if is_train:
        df_cleaned = df_cleaned[df_cleaned['GrLivArea'] < 4000]
    
    return df_cleaned

def log_transform_target(target: pd.Series) -> pd.Series:
    return np.log1p(target)

import pandas as pd

def impute_structural_nan(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()
    
    none_cols = [
        'PoolQC', 'Alley', 'MiscFeature', 'Fence', 'FireplaceQu', 
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in none_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("None")

    zero_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
        'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in zero_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
            
    return df_clean


def impute_statistical_nan(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:

    X_train = train_df.copy()
    X_test = test_df.copy()
 
    mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
    
    for col in mode_cols:
        if col in X_train.columns:
            train_mode = X_train[col].mode()[0]
            
            X_train[col] = X_train[col].fillna(train_mode)
            X_test[col] = X_test[col].fillna(train_mode)
            

    lot_frontage_medians = X_train.groupby('Neighborhood')['LotFrontage'].median().to_dict()
    global_median = X_train['LotFrontage'].median()
    
    for df in [X_train, X_test]:
        if 'LotFrontage' in df.columns:
            df['LotFrontage'] = df.apply(
                lambda row: lot_frontage_medians.get(row['Neighborhood'], global_median) 
                if pd.isna(row['LotFrontage']) else row['LotFrontage'], 
                axis=1
            )
            
    return X_train, X_test