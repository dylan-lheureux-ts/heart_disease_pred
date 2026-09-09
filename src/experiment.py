import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import sys
import json
import os
from pathlib import Path
import yaml
import pickle

if __package__:
    from .preprocessing import (
        clean_data,
        encode_categoricals,
        validate_dataframe,
    )
else:
    from preprocessing import clean_data, encode_categoricals, validate_dataframe

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ─── Configuration ───────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yml"


def load_config(config_path=CONFIG_PATH):
    """Load experiment settings from the YAML configuration file."""
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must contain a YAML mapping: {config_path}"
        )

    config.pop("name", None)
    return config


config = load_config()


def load_and_prepare_data(config):
    """Load the heart disease prediction dataset and prepare it for training."""

    url = Path(__file__).resolve(
    ).parents[1] / "data" / "heart_disease_uci.csv"

    df = pd.read_csv(url)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Convert num to binary:
    # 0 = No heart disease
    # 1, 2, 3, 4 = Heart disease
    df["num"] = (df["num"] > 0).astype(int)

    # Drop any user-specified columns
    if config["features_to_drop"]:
        df = df.drop(columns=config["features_to_drop"], errors="ignore")
        print(f"Dropped features: {config['features_to_drop']}")

    target_column = config.get("target_column", "num")
    validate_dataframe(df, [], target_column)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_column]
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [
        col for col in categorical_cols if col != target_column
    ]

    before = len(df)
    df = clean_data(
        df,
        numeric_cols,
        categorical_cols,
        missing_strategy=config["handle_missing"],
    )
    if config["handle_missing"] == "drop":
        print(f"Dropped rows with missing values: {before} -> {len(df)}")
    else:
        print("Filled missing values with median and categorical mode")

    df = encode_categoricals(df, categorical_cols)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y, len(df), numeric_cols, categorical_cols


def build_model(config):
    """Create a model based on the config."""

    if config["model_type"] == "logistic_regression":
        return LogisticRegression(
            C=config["lr_C"],
            random_state=config["random_state"],
            max_iter=1000
        )
    elif config["model_type"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["rf_n_estimators"],
            max_depth=config["rf_max_depth"],
            random_state=config["random_state"]
        )
    elif config["model_type"] == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["gb_n_estimators"],
            learning_rate=config["gb_learning_rate"],
            max_depth=config["gb_max_depth"],
            random_state=config["random_state"]
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")


def run_experiment(config):
    """Run a single experiment with the given config, tracked by MLflow."""

    # Set the experiment name so all runs are grouped together
    mlflow.set_experiment("heart-disease-prediction")

    # Start an MLflow run
    with mlflow.start_run():

        # ── Log all configuration as parameters ──
        mlflow.log_param("model_type", config["model_type"])
        mlflow.log_param("test_size", config["test_size"])
        mlflow.log_param("random_state", config["random_state"])
        mlflow.log_param("handle_missing", config["handle_missing"])
        mlflow.log_param("scale_features", config["scale_features"])
        mlflow.log_param("metric_average", config["metric_average"])
        mlflow.log_param("features_dropped", str(config["features_to_drop"]))

        # Log model-specific hyperparameters based on model type
        if config["model_type"] == "logistic_regression":
            mlflow.log_param("C", config["lr_C"])
        elif config["model_type"] == "random_forest":
            mlflow.log_param("n_estimators", config["rf_n_estimators"])
            mlflow.log_param("max_depth", str(config["rf_max_depth"]))
        elif config["model_type"] == "gradient_boosting":
            mlflow.log_param("n_estimators", config["gb_n_estimators"])
            mlflow.log_param("learning_rate", config["gb_learning_rate"])
            mlflow.log_param("max_depth", config["gb_max_depth"])

        # ── Load and prepare data ──
        X, y, n_rows, numeric_cols, categorical_cols = load_and_prepare_data(
            config)

        mlflow.log_param("n_rows", n_rows)
        mlflow.log_param("n_features", X.shape[1])

        # ── Split data ──
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=y
        )

        # ── Optionally scale features ──
        if config["scale_features"]:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(
                X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(
                X_test), columns=X_test.columns)

        # ── Train ──
        model = build_model(config)
        print(f"\nTraining {config['model_type']}...")
        model.fit(X_train, y_train)

        # ── Evaluate ──
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        class_count = len(model.classes_)
        metric_average = config["metric_average"]
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": X_train.shape[1],
        }
        accuracy = accuracy_score(y_test, y_pred)
        if class_count == 2:
            precision = precision_score(
                y_test, y_pred, average=metric_average, zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, average=metric_average, zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred, average=metric_average, zero_division=0
            )
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            precision = precision_score(
                y_test, y_pred, average=metric_average, zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, average=metric_average, zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred, average=metric_average, zero_division=0
            )
            auc = roc_auc_score(
                y_test,
                y_prob,
                multi_class="ovr",
                average=metric_average,
            )

        # ── Log metrics ──
        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("precision", round(precision, 4))
        mlflow.log_metric("recall", round(recall, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("auc_roc", round(auc, 4))
        # Check thresholds
        if metrics["accuracy"] < config["min_accuracy"]:
            print(
                f"\nWARNING: Accuracy {metrics['accuracy']} is below threshold {config['min_accuracy']}")
        if metrics["f1_score"] < config["min_f1"]:
            print(
                f"\nWARNING: F1 {metrics['f1_score']} is below threshold {config['min_f1']}")

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {model_path}")

        # Save metrics
        os.makedirs("metrics", exist_ok=True)
        metrics_path = "metrics/results.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

        # Exit with error if thresholds not met
        if metrics["accuracy"] < config["min_accuracy"]:
            print(f"\nFAILED: Accuracy below threshold")
            sys.exit(1)
        if metrics["f1_score"] < config["min_f1"]:
            print(f"\nFAILED: F1 score below threshold")
            sys.exit(1)

        print("\nAll thresholds passed!")

        # ── Log the trained model as an artifact ──
        mlflow.sklearn.log_model(model, "model")

        # ── Log the config file as an artifact for reference ──
        config_path = "config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        mlflow.log_artifact(config_path)
        os.remove(config_path)  # clean up temp file

        # ── Print results ──
        print(f"\n{'='*50}")
        print(f"Model:     {config['model_type']}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print(f"{'='*50}")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print("View this run in the UI: mlflow ui")

    return run_id


if __name__ == "__main__":
    run_experiment(config)
