import numpy as np 
import pandas as pd 

def remove_outliers(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df_cleaned = df.copy()
    if is_train and 'GrLivArea' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['GrLivArea'] < 4000]
    return df_cleaned

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

def impute_statistical_nan(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Trains statistics on X_train and applies them safely to both frames without leakage."""
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    # 1. Handle general missing categorical/numerical markers
    ignore_cols = ['Id', 'SalePrice']
    all_cols = [c for c in X_train_clean.columns if c not in ignore_cols]
    
    for col in all_cols:
        if X_train_clean[col].dtype in ['float64', 'int64']:
            fill_val = X_train_clean[col].median()
        else:
            fill_val = X_train_clean[col].mode()[0] if not X_train_clean[col].mode().empty else "None"
            
        X_train_clean[col] = X_train_clean[col].fillna(fill_val)
        X_test_clean[col] = X_test_clean[col].fillna(fill_val)
        
    # 2. Localized Neighborhood LotFrontage Imputation
    if 'LotFrontage' in X_train_clean.columns:
        lot_frontage_medians = X_train_clean.groupby('Neighborhood')['LotFrontage'].median().to_dict()
        global_median = X_train_clean['LotFrontage'].median()
        
        X_train_clean['LotFrontage'] = X_train_clean.apply(
            lambda row: lot_frontage_medians.get(row['Neighborhood'], global_median) if pd.isna(row['LotFrontage']) else row['LotFrontage'], axis=1
        )
        X_test_clean['LotFrontage'] = X_test_clean.apply(
            lambda row: lot_frontage_medians.get(row['Neighborhood'], global_median) if pd.isna(row['LotFrontage']) else row['LotFrontage'], axis=1
        )
            
    return X_train_clean, X_test_clean

def get_skewed_features_list(df: pd.DataFrame) -> list:
    """Calculates skewed features on the Training dataset ONLY."""
    ignore_cols = ['Id', 'SalePrice']
    numeric_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c not in ignore_cols]
    skewed_values = df[numeric_cols].skew()
    high_skew_cols = skewed_values[abs(skewed_values) > 0.75].index.tolist()
    return high_skew_cols

def transform_skewed_features(df: pd.DataFrame, skew_cols_list: list) -> pd.DataFrame:
    """Applies log transform to a predetermined, fixed list of features."""
    df_transformed = df.copy()
    for col in skew_cols_list:
        if col in df_transformed.columns:
            df_transformed[col] = np.log1p(df_transformed[col])
    return df_transformed

def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    df_ord = df.copy()
    quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    
    for col in qual_cols:
        if col in df_ord.columns:
            df_ord[col] = df_ord[col].map(quality_map).fillna(0)
            
    return df_ord
