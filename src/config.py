from pathlib import Path

# Automatically resolves the absolute path to the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_TRAIN_DATA_PATH = DATA_DIR / "train.csv"
RAW_TEST_DATA_PATH = DATA_DIR / "test.csv"

# Model Artifacts & Outputs
OUTPUT_DIR = BASE_DIR / "models"

# Dataset Target Variable
TARGET = "SalePrice"

# Global Constants for Reproducibility
RANDOM_STATE = 42
NUM_FOLDS = 5