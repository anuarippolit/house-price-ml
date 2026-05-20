import numpy as np
import pandas as pd
import joblib 
from src.config import RAW_TRAIN_DATA_PATH, RAW_TEST_DATA_PATH, OUTPUT_DIR
from src.preprocessing import (
    remove_outliers, impute_structural_nan, impute_statistical_nan,
    get_skewed_features_list, transform_skewed_features, encode_ordinal_features
)
from src.engineering import engineer_features
from src.models import get_ensemble_blend_pipeline

def run_production_pipeline():

    train_raw = pd.read_csv(RAW_TRAIN_DATA_PATH)
    test_raw = pd.read_csv(RAW_TEST_DATA_PATH)


    X_train = remove_outliers(train_raw, is_train=True)
    y_train_log = np.log1p(X_train['SalePrice'])
    X_train = X_train.drop(columns=['SalePrice'])
    
    X_train = impute_structural_nan(X_train)


    X_test = test_raw.copy()
    X_test = impute_structural_nan(X_test)


    X_train, X_test = impute_statistical_nan(X_train, X_test)


    X_train = encode_ordinal_features(X_train)
    X_test = encode_ordinal_features(X_test)
    
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)


    skewed_features_anchor = get_skewed_features_list(X_train)
    X_train = transform_skewed_features(X_train, skewed_features_anchor)
    X_test = transform_skewed_features(X_test, skewed_features_anchor)


    categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()


    pipeline = get_ensemble_blend_pipeline(categorical_cols, numerical_cols)
    pipeline.fit(X_train, y_train_log)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = OUTPUT_DIR / "final_ensemble_model.pkl"
    joblib.dump(pipeline, model_save_path)
    print(f"Model artifact successfully serialized to {model_save_path}")

    print("Generating predictions")
    log_predictions = pipeline.predict(X_test)
    final_dollar_predictions = np.expm1(log_predictions)

    submission = pd.DataFrame({
        'Id': test_raw['Id'],
        'SalePrice': final_dollar_predictions
    })

    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved to '{submission_path}'\\n")
    print(submission.head())

if __name__ == "__main__":
    run_production_pipeline()