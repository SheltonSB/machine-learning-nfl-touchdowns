# explain_shap.py

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Load model and data
model = joblib.load('../models/qb_td_model.pkl')
data = pd.read_csv('../data/processed/final_dataset.csv')

# Clean: drop NaNs
data = data.dropna()

# Define the same features used during training
feature_cols = [
    'Passing Yards_roll3',
    'TD Passes_roll3',
    'Passes Attempted_roll3',
    'Age',
    'Experience',
    'Height (inches)',
    'Weight (lbs)'
]

X = data[feature_cols]

# Use a subset for speed if dataset is large
X_sample = X.sample(200, random_state=42)

# Create SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_sample)

# Global explanation (what matters most in general)
print("Generating SHAP summary plot...")
shap.plots.beeswarm(shap_values, max_display=10)

# Optional: save to file
plt.savefig("../models/shap_summary.png", dpi=300)
print(" SHAP summary saved to models/shap_summary.png")


