import pandas as pd 

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()

    df_feat['TotalSF'] = df_feat['1stFlrSF'] + df_feat['2ndFlrSF'] + df_feat['TotalBsmtSF']
    df_feat['TotalBath'] = (df_feat['FullBath'] + (0.5 * df_feat['HalfBath']) + df_feat['BsmtFullBath'] + (0.5 * df_feat['BsmtHalfBath']))

    df_feat['AgeAtSale'] = df_feat['YrSold'] - df_feat['YearBuilt']
    df_feat['YearsSinceRemodel'] = df_feat['YrSold'] - df_feat['YearRemodAdd']
    df_feat['AgeAtSale'] = df_feat['AgeAtSale'].apply(lambda x: max(0, x))
    df_feat['YearsSinceRemodel'] = df_feat['YearsSinceRemodel'].apply(lambda x: max(0, x))

    df_feat['IsRemodeled'] = (df_feat['YearBuilt'] != df_feat['YearRemodAdd']).astype(int)
    df_feat['HasWoodDeck'] = (df_feat['WoodDeckSF'] > 0).astype(int)
    df_feat['HasOpenPorch'] = (df_feat['OpenPorchSF'] > 0).astype(int)
    df_feat['HasEnclosedPorch'] = (df_feat['EnclosedPorch'] > 0).astype(int)
    df_feat['HasPool'] = (df_feat['PoolArea'] > 0).astype(int)

    return df_feat
