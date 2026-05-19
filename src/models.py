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