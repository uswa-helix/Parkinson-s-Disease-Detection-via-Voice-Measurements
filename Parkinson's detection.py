import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    make_scorer,
    recall_score
)
import joblib
import shap  # ADDED: Import the SHAP library

# ==========================================
# 1. Data Loading, Group Extraction & Preparation
# ==========================================
data = pd.read_csv("parkinsons.csv")

# Extract the patient ID from the 'name' column (e.g., 'phon_R01_S01_1' -> 'S01')
groups = data['name'].apply(lambda x: x.split('_')[2])

# Drop the non-predictive columns
X = data.drop(columns=['name', 'status'])
y = data['status']
feature_names = X.columns
target_names = ['Healthy', "Parkinson's"]

# Split the data while grouping by Patient ID
# This guarantees that a patient's recordings are entirely in Train OR entirely in Test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
groups_tr = groups.iloc[train_idx]  # We need the training groups later for cross-validation

yte_array = yte.values 

# ==========================================
# 2. Baseline Model & Feature Importance
# ==========================================
# Removed Scaler and added class_weight='balanced' to match the pipeline structure
base_model = RandomForestClassifier(class_weight='balanced', random_state=42)
base_model.fit(Xtr, ytr)

print("--- Top 5 Diagnostic Voice Features ---")
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': base_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance_df.head(5).to_string(index=False))
print("\n")

# ==========================================
# 3. Diagnosing Baseline False Negatives
# ==========================================
print("--- Diagnosing Baseline Missed Parkinson's Cases ---")
y_probs_base = base_model.predict_proba(Xte)

for i in range(len(yte_array)):
    true_class = "Parkinson's" if yte_array[i] == 1 else "Healthy"
    prob_parkinsons = y_probs_base[i, 1] 

    if yte_array[i] == 1 and prob_parkinsons < 0.50:
        print(f"Sample {i}: True Label = {true_class} | Model assigned Parkinson's Prob: {prob_parkinsons:.2%}")
print("\n")

# ==========================================
# 4. Optimized Clinical Model (Pipeline & GridSearchCV)
# ==========================================
print("--- Tuning and Training Optimized Clinical Model ---")

# Removed StandardScaler since Random Forests are scale-invariant
pipeline = Pipeline([
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

param_grid = {
    'rf__n_estimators': [50, 100, 200, 300],
    'rf__max_depth': [None, 5, 10, 20],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Use StratifiedGroupKFold to prevent data leakage during Cross-Validation
cv_strategy = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

parkinsons_recall = make_scorer(recall_score, pos_label=1)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring=parkinsons_recall,  
    n_jobs=-1,                  
    verbose=1
)

# Fit the GridSearch, passing 'groups_tr' to ensure folds respect patient boundaries
grid_search.fit(Xtr, ytr, groups=groups_tr)

print(f"\nBest Hyperparameters Found:")
for param, value in grid_search.best_params_.items():
    print(f" - {param}: {value}")
print("\n")

final_model = grid_search.best_estimator_
final_preds = final_model.predict(Xte)

# ==========================================
# 5. Final Evaluation Metrics & Visualization
# ==========================================
print(f"Final Model Overall Accuracy: {final_model.score(Xte, yte):.4f}\n")
print("--- Final Classification Report (Default 50% Threshold) ---")
print(classification_report(yte, final_preds, target_names=target_names, zero_division=0.0))

cm_final = confusion_matrix(yte, final_preds)
disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=target_names)
disp_final.plot(cmap=plt.cm.Purples)
plt.title("Patient-Grouped Confusion Matrix (50% Threshold)")
plt.show()

# ==========================================
# 6. Clinical Threshold Adjustment (30%) & ROC Visualization
# ==========================================
print("\n--- Applying Custom Clinical Threshold (30%) ---")

final_probs = final_model.predict_proba(Xte)
parkinsons_probs = final_probs[:, 1]

custom_threshold = 0.30
clinical_preds = np.where(parkinsons_probs >= custom_threshold, 1, 0)

print("--- Clinical Classification Report (30% Threshold) ---")
print(classification_report(yte, clinical_preds, target_names=target_names, zero_division=0.0))

cm_clinical = confusion_matrix(yte, clinical_preds)
disp_clinical = ConfusionMatrixDisplay(confusion_matrix=cm_clinical, display_labels=target_names)
disp_clinical.plot(cmap=plt.cm.Reds)
plt.title("Patient-Grouped Confusion Matrix (30% Threshold)")
plt.show()

print("\n--- Generating ROC Curve ---")
roc_display = RocCurveDisplay.from_predictions(
    yte,
    parkinsons_probs,
    pos_label=1,
    name="Optimized RF",
    color="darkorange"
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--") 
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.show()

# ==========================================
# 7. Model Explainability with SHAP 
# ==========================================
print("\n--- Generating SHAP Explanations ---")

# Extract the fitted Random Forest classifier from the pipeline
rf_model = final_model.named_steps['rf']

# Initialize the SHAP TreeExplainer
explainer = shap.TreeExplainer(rf_model)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(Xte, check_additivity=False)

# For classification, Random Forest explainers output a list or 3D array of values for each class. 
# We want to explain the predictions for Class 1 (Parkinson's).
if isinstance(shap_values, list):
    shap_values_parkinsons = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_parkinsons = shap_values[:, :, 1]
else:
    shap_values_parkinsons = shap_values

# Generate the SHAP Summary Plot
plt.figure()
plt.title("SHAP Summary Plot (Impact on Parkinson's Prediction)")
shap.summary_plot(shap_values_parkinsons, Xte, show=False)
plt.tight_layout()
plt.show()

# ==========================================
# 8. Save the Model
# ==========================================
joblib.dump(final_model, 'clinical_rf_pipeline_parkinsons.pkl')
print("Model successfully saved!")
