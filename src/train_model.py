"""TensorFlow training script for the NFL QB Touchdown Predictor."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


def get_project_paths() -> Dict[str, Path]:
    """Return key project paths."""

    root = Path(__file__).resolve().parent.parent
    return {
        "root": root,
        "data": root / "data" / "processed" / "final_dataset.csv",
        "models": root / "models",
        "scaler": root / "models" / "feature_scaler.pkl",
        "model": root / "models" / "qb_td_model.keras",
        "metrics": root / "models" / "training_metrics.json",
        "classification": root / "models" / "classification_report.txt",
    }


def load_dataset(dataset_path: Path, target_col: str) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the processed dataset and return features, labels, and column names."""

    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {dataset_path}")

    data = pd.read_csv(dataset_path)

    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # Keep only numeric features
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_col]

    if not feature_cols:
        raise ValueError("No numeric feature columns available for training")

    X = data[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y = data[target_col].astype(int).to_numpy(dtype=np.int32)

    return X, y, feature_cols


def build_model(input_dim: int) -> tf.keras.Model:
    """Create a simple feedforward neural network for binary classification."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )

    return model


def compute_fold_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics for a validation fold."""

    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def main() -> int:
    """Train a TensorFlow model and save artifacts/metrics."""

    tf.keras.utils.set_random_seed(42)

    paths = get_project_paths()
    paths["models"].mkdir(parents=True, exist_ok=True)

    X, y, feature_cols = load_dataset(paths["data"], target_col="threw_td")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, paths["scaler"])

    # Cross-validation for robust metrics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train), start=1):
        model = build_model(X_train_scaled.shape[1])
        history = model.fit(
            X_train_scaled[train_idx],
            y_train[train_idx],
            validation_data=(X_train_scaled[val_idx], y_train[val_idx]),
            epochs=100,
            batch_size=64,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                )
            ],
        )

        y_val_prob = model.predict(X_train_scaled[val_idx], verbose=0).ravel()
        metrics = compute_fold_metrics(y_train[val_idx], y_val_prob)
        metrics["best_val_loss"] = float(min(history.history["val_loss"]))
        metrics["epochs_trained"] = len(history.history["loss"])
        metrics["fold"] = fold
        fold_metrics.append(metrics)

    avg_metrics = {
        metric: float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    # Train final model on full training split
    final_model = build_model(X_train_scaled.shape[1])
    final_history = final_model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=64,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            ),
        ],
    )

    y_test_prob = final_model.predict(X_test_scaled, verbose=0).ravel()
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": f1_score(y_test, y_test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_test_prob),
    }

    report = classification_report(y_test, y_test_pred)

    final_model.save(paths["model"])

    metrics_payload = {
        "feature_columns": feature_cols,
        "cv_folds": fold_metrics,
        "cv_average": avg_metrics,
        "test_metrics": test_metrics,
        "history": {
            "loss": [float(x) for x in final_history.history["loss"]],
            "val_loss": [float(x) for x in final_history.history["val_loss"]],
            "accuracy": [float(x) for x in final_history.history.get("accuracy", [])],
            "val_accuracy": [
                float(x) for x in final_history.history.get("val_accuracy", [])
            ],
        },
    }

    with open(paths["metrics"], "w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    with open(paths["classification"], "w", encoding="utf-8") as fp:
        fp.write(report)

    print("\n=== TensorFlow QB Touchdown Model ===")
    print("Saved model:", paths["model"].relative_to(paths["root"]))
    print("Saved scaler:", paths["scaler"].relative_to(paths["root"]))
    print("Saved metrics:", paths["metrics"].relative_to(paths["root"]))
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.3f}")

    print("\nClassification Report:\n")
    print(report)

    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
