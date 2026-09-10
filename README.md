# Heart Disease Prediction MLOps Pipeline

An end-to-end machine learning project for predicting heart disease from the UCI Heart Disease dataset. The project combines data validation and preprocessing, model training and evaluation, MLflow experiment tracking, hyperparameter experimentation, DVC data versioning, automated testing, GitHub Actions CI, and Evidently-based production drift monitoring.

## Project Overview

The pipeline converts the original `num` target into a binary classification target:

- `0` = no heart disease
- `1` = heart disease

The training workflow loads the dataset from `data/heart_disease_uci.csv`, removes configured columns, handles missing values, one-hot encodes categorical features, optionally scales numeric features, trains a selected scikit-learn model, evaluates it, logs the run to MLflow, and saves the trained model and evaluation metrics.

The default configuration uses Gradient Boosting with a 20% test split, median missing-value imputation, feature scaling, and minimum accuracy/F1 thresholds of 0.70.

## Technologies

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- PyYAML
- MLflow
- DVC
- Evidently
- Pytest
- GitHub Actions

## Project Structure

```text
heart_disease_pred/
├── .dvc/                         # DVC configuration
├── .github/
│   └── workflows/
│       └── heart-disease-pred.yml # CI pipeline
├── configs/
│   └── config.yml                # Training configuration
├── data/
│   └── heart_disease_uci.csv.dvc  # DVC pointer for source dataset
├── drift_monitoring/
│   ├── detect_drift.py            # Overall Evidently drift detection
│   ├── drift_over_time.py         # Tracks age distribution over time
│   ├── simulate_drift.py          # Generates simulated production data
│   ├── targeted_drift.py          # Critical-feature drift analysis
│   └── reports/                   # Drift datasets and generated reports
├── metrics/
│   └── results.json               # Model evaluation metrics
├── mlruns/                        # Local MLflow tracking data
├── models/
│   └── model.pkl                  # Saved trained model
├── src/
│   ├── analyze.py                 # Analyze MLflow runs
│   ├── experiment.py              # Main training/evaluation script
│   ├── monitor_drift.py           # Drift monitoring orchestrator
│   ├── preprocessing.py           # Data validation and preprocessing
│   └── sweep.py                   # Run multiple model configurations
├── tests/
│   └── test_preprocessing.py      # Unit, data, and model validation tests
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/dylan-lheureux-ts/heart_disease_pred.git
cd heart_disease_pred
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents the activation script from running, activate the environment from another supported shell or adjust the local PowerShell execution policy as appropriate for your system.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository pins the main dependencies in `requirements.txt` to keep the environment reproducible.

### 4. Retrieve DVC-managed data

The source dataset and simulated drift datasets are versioned with DVC. The repository's configured DVC remote uses Amazon S3.

If you have access to the configured remote, authenticate with your AWS credentials and then run:

```powershell
$env:AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
$env:AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
$env:AWS_DEFAULT_REGION="us-east-1"

dvc pull
```

Do not commit AWS credentials or other secrets to the repository.

After `dvc pull`, verify the dataset is available at:

```text
data/heart_disease_uci.csv
```

## Configuration

Training settings are stored in `configs/config.yml`.

The current configuration includes:

```yaml
model_type: gradient_boosting
target_column: num
metric_average: weighted
test_size: 0.2
random_state: 42
handle_missing: median
scale_features: true
features_to_drop: [id, dataset]
```

Supported model types are:

- `logistic_regression`
- `random_forest`
- `gradient_boosting`

Model-specific hyperparameters and minimum performance thresholds are also defined in the configuration file.

## Training the Model

Run training from the repository root:

```powershell
python src/experiment.py
```

The training script:

1. Loads the DVC-managed heart disease dataset.
2. Converts `num` to a binary target.
3. Removes configured columns such as `id` and `dataset`.
4. Handles missing values using the configured strategy.
5. One-hot encodes categorical variables.
6. Splits the data into training and test sets.
7. Optionally standardizes features.
8. Trains the configured scikit-learn model.
9. Calculates accuracy, precision, recall, F1 score, and AUC-ROC.
10. Checks the configured minimum accuracy and F1 thresholds.
11. Saves the trained model to `models/model.pkl`.
12. Saves evaluation metrics to `metrics/results.json`.
13. Logs parameters, metrics, the model, and a configuration snapshot to MLflow.

If the configured accuracy or F1 threshold is not met, the training process exits with code `1`.

### Run a Hyperparameter Sweep

The sweep script runs nine predefined model configurations covering Logistic Regression, Random Forest, and Gradient Boosting:

```powershell
python src/sweep.py
```

The experiments are logged to MLflow so that their results can be compared.

### Analyze MLflow Results

After running experiments or the sweep:

```powershell
python src/analyze.py
```

This reports the top five completed runs by F1 score and summarizes F1 performance by model type.

## MLflow

The training script uses the MLflow experiment:

```text
heart-disease-prediction
```

Start the MLflow UI from the repository root with:

```powershell
mlflow ui
```

Then open the local MLflow address shown in the terminal to compare runs, parameters, metrics, and logged models.

## Drift Monitoring

The project includes an Evidently-based monitoring pipeline that compares the reference training data against simulated production data for three months.

### Generate simulated production data

From the repository root:

```powershell
python drift_monitoring/simulate_drift.py
```

This creates:

```text
drift_monitoring/reports/reference_data.csv
drift_monitoring/reports/month1_data.csv
drift_monitoring/reports/month2_data.csv
drift_monitoring/reports/month3_data.csv
```

Month 1 represents normal production data. Month 2 introduces moderate shifts in cholesterol and resting blood pressure. Month 3 introduces stronger shifts in several features, including age, cholesterol, resting blood pressure, maximum heart rate, ST depression, and exercise-related variables.

### Run the complete monitoring pipeline

```powershell
python src/monitor_drift.py
```

The monitoring pipeline performs three analyses:

1. **Overall dataset drift** across 13 input features.
2. **Drift over time** for the `age` feature.
3. **Targeted drift analysis** for critical features including `sex`, `ca`, `chol`, `thalch`, and `age`.

Evidently uses a feature-level p-value threshold of `0.05`. The overall dataset drift threshold is `40%` of monitored features.

The pipeline generates HTML reports in:

```text
drift_monitoring/reports/
```

and writes the age drift timeline to:

```text
drift_monitoring/reports/drift_timeline.json
```

### Drift exit codes

The monitoring pipeline is designed to return:

- **Exit code 0** when overall dataset drift remains below the configured threshold.
- **Exit code 1** when the overall drift threshold is exceeded.

With the current simulated data, Month 3 is intentionally designed to exceed the threshold, so an exit code of `1` is expected when the complete monitoring pipeline is run.

The recommended response to significant drift is:

```text
Detect drift → Investigate cause → Evaluate model performance → Retrain if performance has degraded
```

## Testing

Run the complete test suite from the repository root:

```powershell
python -m pytest tests/ -v --tb=short
```

The tests cover:

- DataFrame validation
- Missing required columns
- Missing target columns
- Empty DataFrames
- Numeric missing-value imputation
- Categorical missing-value imputation
- Median replacement
- Dropping rows with missing values
- Protection of the original DataFrame during preprocessing
- One-hot encoding
- Preservation of row counts during encoding
- Data-quality metrics
- Expected dataset columns
- Target-value validation
- Numeric feature ranges
- Model prediction shape and type
- Minimum model accuracy

## Continuous Integration

GitHub Actions runs the project's automated pipeline on pushes and pull requests to `master`, as well as manual workflow runs.

The CI workflow:

1. Checks out the repository.
2. Sets up Python 3.12.
3. Installs the dependencies.
4. Authenticates to the configured DVC storage.
5. Pulls DVC-managed data.
6. Runs the Pytest test suite.
7. Trains the model after tests pass.
8. Uploads training metrics as a workflow artifact.
9. Uploads the trained model as a workflow artifact when training succeeds.

This creates a basic MLOps workflow in which tests must pass before model training proceeds.

## Outputs

After a successful training run, the primary local outputs are:

```text
models/model.pkl
metrics/results.json
mlruns/
```

The drift-monitoring workflow produces HTML reports and a JSON timeline under:

```text
drift_monitoring/reports/
```

## Reproducibility

The project uses several mechanisms to improve reproducibility:

- Pinned Python package versions in `requirements.txt`
- YAML-based training configuration
- Fixed `random_state` values for model training and dataset splitting
- DVC for dataset versioning
- MLflow for experiment tracking
- Automated Pytest validation
- GitHub Actions for continuous integration

## Important Note

This project is intended for machine learning and MLOps portfolio/educational purposes. It is not a clinical diagnostic system and should not be used to make medical decisions.

# Drift Monitoring Analysis

## Overview

Drift monitoring was performed by comparing the reference dataset used during model development with simulated Month 3 production data. Evidently was used to evaluate whether the distribution of the model's input features had changed after deployment.
The monitoring pipeline evaluates 13 input features and excludes the target variable, `num`, from feature drift calculations.
The overall dataset drift threshold was configured at 40%. If at least 40% of monitored features show statistically significant drift, the monitoring process flags the dataset for further investigation.

## Which features showed drift and why?

Month 3 showed drift in six of the 13 monitored features:
* `age`
* `trestbps`
* `chol`
* `thalch`
* `oldpeak`
* `ca`

The drift was primarily caused by the simulated production-data changes introduced during the monitoring experiment.
For example, Month 3 intentionally increased cholesterol (`chol`) and resting blood pressure (`trestbps`). The age distribution was also shifted toward older patients for a portion of the production data. Additional changes were introduced to maximum heart rate (`thalch`), ST depression (`oldpeak`), and the number of major vessels (`ca`)
These changes simulate a realistic situation in which the population or characteristics of patients being evaluated after deployment differ from the population represented in the original training data.
Overall, 6 out of 13 features showed statistically significant drift, resulting in an overall drift share of 46.2%.

## Would this drift likely affect model performance?

Yes, the observed drift could affect model performance.
Several of the features showing drift are clinically meaningful predictors of heart disease. Changes in variables such as age, cholesterol, resting blood pressure, maximum heart rate, and ST depression could change the relationship between the input data and the model's predictions.
Even if the model's accuracy has not yet decreased, substantial changes in the input distribution can make predictions less reliable because the model is receiving data that differs from the distribution it learned during training.
The fact that 46.2% of monitored features showed drift is significant because it exceeds the configured 40% dataset-level threshold.

## Recommended action

I would recommend investigating the drift and preparing for retraining rather than immediately retraining the model.
First, the drift should be investigated to determine whether the changes represent a genuine change in the patient population, a data collection or preprocessing issue, or another operational change.
Model performance metrics should then be evaluated on recent production data. If performance has degraded alongside the observed feature drift, the model should be retrained using representative and recent data.
If model performance remains stable, monitoring should continue while the cause and persistence of the drift are investigated.
Therefore, the recommended workflow is:

**Detect drift → Investigate cause → Evaluate model performance → Retrain if performance has degraded.**
