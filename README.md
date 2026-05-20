# House Price Prediction

The project's goal is to predict house pricing according to dataset on Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

The stages of production: 
* EDA -> in notebooks 
* Preprocessing -> all functions are in preprocessing.py 
* Training and tunning 

This pipeline achieved the accuray of 0.12418. For the future consideration is to change the weights of each model in ensemble voting. 
---

## Directory & File Architecture

### 1. config.py
The centralized configuration engine for the entire ecosystem. It establishes project constants, environment-agnostic absolute paths, and reproducibility variables.
* **Key Components:** Resolves absolute directory patterns using Python's `pathlib.Path`, maps paths for raw files (`train.csv`, `test.csv`), handles global `RANDOM_STATE` parameters, and configures cross-validation fold numbers.

### 2. preprocessing.py
The data sanitization and matrix cleansing layer. This file contains stateless deterministic functions tailored to handle structured anomalies, statistical omissions, structural deficiencies, and geometric skew distributions.
* **Key Components:**
  * `remove_outliers()`: Hard-filters mathematical leverage points (such as properties with excessive square footage but deflated prices) strictly on training subsets.
  * `impute_structural_nan()`: Explicitly maps categorical `NaN` flags to string literals (`"None"`) and matching numerical attributes to `0` where missing values represent structural absence rather than data collection omissions.
  * `impute_statistical_nan()`: Dynamically imputes true missing records using localized parameters (such as computing neighborhood-specific median street frontage metrics) extracted solely from the training data.
  * `get_skewed_features_list()` & `transform_skewed_features()`: Evaluates feature-level data distributions. Anchors high-skew feature lists based on the training slice to map matching log-transform adjustments onto unseen data matrices uniformly.
  * `encode_ordinal_features()`: Maps categorical quality matrices (*Po, Fa, TA, Gd, Ex*) to direct integer gradients (`1` to `5`) to let downstream estimators evaluate explicit feature hierarchies.

### 3. engineering.py
The logic synthesis domain. It processes individual continuous and dimensional records to build domain-specific indicators that improve pattern recognition.
* **Key Components:** Collapses multi-collinear volumetric measures into unified interaction measurements (such as `TotalSF` and `TotalBath`), transforms chronological milestones into relative duration values (`AgeAtSale`), and flags specific structural upgrades via binary vector switches.

### 4. models.py
The ensemble orchestration layer. It handles preprocessing routing and encapsulates complex model combinations inside end-to-end processing pipelines.
* **Key Components:** Combines `StandardScaler` and `OneHotEncoder` tracking targets within an automated Scikit-Learn `ColumnTransformer`. It houses optimized architectures for Ridge Regression, XGBoost, and LightGBM inside a weighted `VotingRegressor` blend, alongside automated cross-validation performance evaluation hooks (`evaluate_pipeline()`).

---

## Production Entry Point: main.py

`main.py` functions as the absolute structural orchestrator that imports modular scripts and executes the entire data pipeline end-to-end via a single terminal execution block. It serves as the transition from an experimental scratchpad environment to production-ready software automation.

### Production Execution Pipeline
1. **Ingest Data:** Automatically loads the source datasets via paths configured in `config.py`.
2. **Execute Custom Transformations:** Runs training data through outlier cleansing, structural adjustments, and captures statistical and skew properties.
3. **Align Test Sets:** Transforms testing frames utilizing the explicit metrics anchored by the training matrices to guarantee 0% data leakage.
4. **Train and Serialize:** Fits the complete ensemble voting pipeline on 100% of the training features and serializes the state to a binary file (`final_ensemble_model.pkl`) using `joblib` inside the models directory.
5. **Inference & Export:** Scores the test partition, reverses target scale modifications (`np.expm1`), and formats the outputs directly into a clean submission file.