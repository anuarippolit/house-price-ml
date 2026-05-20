import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

def get_baseline_pipeline(categorical_cols : list, numerical_cols : list) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=10.0))  # A stable, regularized linear baseline
    ])

    return pipeline

def evaluate_pipeline(X: pd.DataFrame, y: pd.DataFrame, pipeline: Pipeline) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)

    print(f"--- 5-Fold CV Results ---")
    print(f"Folds RMSE: {np.round(rmse_scores, 4)}")
    print(f"Mean RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
    
    return rmse_scores.mean()

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_xgboost_pipeline(categorical_cols: list, numerical_cols: list) -> Pipeline:
    
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])
    return pipeline

def get_lightgbm_pipeline(categorical_cols: list, numerical_cols: list) -> Pipeline:

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1  # Suppress internal logging clutter
        ))
    ])
    return pipeline

from sklearn.ensemble import VotingRegressor

def get_ensemble_blend_pipeline(categorical_cols: list, numerical_cols: list) -> Pipeline:

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    preprocessor.set_output(transform="pandas")
    
    ridge_model = Ridge(alpha=10.0)
    
    xgb_model = XGBRegressor(
        colsample_bytree=0.7,
        learning_rate=0.03,
        max_depth=3,
        n_estimators=500,
        reg_alpha=0,
        reg_lambda=2,
        subsample=0.9,
        random_state=42,
        n_jobs=-1
    )
    
    lgbm_model = LGBMRegressor(
        colsample_bytree=0.7,
        learning_rate=0.03,
        max_depth=4,
        n_estimators=500,
        num_leaves=127,
        reg_alpha=0,
        reg_lambda=5,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    ensemble_voting = VotingRegressor(
        estimators=[
            ('ridge', ridge_model),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        weights=[1, 1, 1]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', ensemble_voting)
    ])
    
    return pipeline