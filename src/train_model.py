# train_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Load preprocessed data
data = pd.read_csv('../data/processed/final_dataset.csv')

# Drop rows with any missing values (optional for simplicity)
data = data.dropna()

# Define features to use
feature_cols = [
    'Passing Yards_roll3',
    'TD Passes_roll3',
    'Passes Attempted_roll3',
    'Age',
    'Experience',
    'Height (inches)',
    'Weight (lbs)'
]

# Target column
target_col = 'threw_td'

# Split data
X = data[feature_cols]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print("Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("F1 Score:", round(f1_score(y_test, y_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/qb_td_model.pkl')
print(" Model saved to models/qb_td_model.pkl")
