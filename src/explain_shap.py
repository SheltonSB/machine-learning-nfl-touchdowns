"""Generate SHAP explanations for the TensorFlow touchdown model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf


def get_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    return {
        "root": root,
        "data": root / "data" / "processed" / "final_dataset.csv",
        "model": root / "models" / "qb_td_model.keras",
        "scaler": root / "models" / "feature_scaler.pkl",
        "metrics": root / "models" / "training_metrics.json",
        "output": root / "models" / "shap_summary.png",
    }


def load_feature_columns(metrics_path: Path) -> list[str] | None:
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    return payload.get("feature_columns")


def main() -> int:
    paths = get_paths()

    if not paths["model"].exists():
        raise FileNotFoundError(
            f"TensorFlow model not found at {paths['model']}. Train the model first."
        )

    if not paths["scaler"].exists():
        raise FileNotFoundError(
            f"Feature scaler not found at {paths['scaler']}. Train the model first."
        )

    if not paths["data"].exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {paths['data']}. Run preprocessing first."
        )

    feature_columns = load_feature_columns(paths["metrics"])

    data = pd.read_csv(paths["data"])

    if feature_columns is None:
        numeric_cols = data.select_dtypes(include=[float, int]).columns.tolist()
        feature_columns = [col for col in numeric_cols if col != "threw_td"]

    X = data[feature_columns].fillna(0)
    scaler = joblib.load(paths["scaler"])
    X_scaled = scaler.transform(X.to_numpy())

    # Use DataFrame for nicer feature names downstream
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

    model = tf.keras.models.load_model(paths["model"], compile=False)

    sample_background = shap.sample(
        X_scaled_df, min(100, len(X_scaled_df)), random_state=42
    )
    sample_eval = shap.sample(X_scaled_df, min(200, len(X_scaled_df)), random_state=42)

    def model_predict(batch):
        if isinstance(batch, pd.DataFrame):
            array = batch.to_numpy(dtype=np.float32)
        else:
            array = np.asarray(batch, dtype=np.float32)
        return model.predict(array, verbose=0)

    explainer = shap.Explainer(model_predict, sample_background)
    shap_values = explainer(sample_eval)

    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(paths["output"], dpi=300)

    print("SHAP summary saved to", paths["output"].relative_to(paths["root"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
