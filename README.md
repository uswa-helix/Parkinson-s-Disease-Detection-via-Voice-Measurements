# Parkinson's Disease Detection via Voice Features 

This repository contains a machine learning pipeline designed to detect Parkinson's Disease using biomedical acoustic voice measurements. It utilizes the **UCI Oxford Parkinson's Disease dataset** and emphasizes clinically sound machine learning practices, including patient-level cross-validation and sensitivity optimization.

##  Table of Contents

* [Project Overview](https://www.google.com/search?q=%23project-overview)
* [Dataset](https://www.google.com/search?q=%23dataset)
* [Methodology & Key Features](https://www.google.com/search?q=%23methodology--key-features)
* [Results & Clinical Relevance](https://www.google.com/search?q=%23results--clinical-relevance)
* [Requirements](https://www.google.com/search?q=%23requirements)
* [Usage](https://www.google.com/search?q=%23usage)

---

##  Project Overview

Diagnosing Parkinson's Disease early and accurately can significantly improve patient outcomes. Voice degradation is often one of the earliest indicators of the disease. This script trains an optimized Random Forest classifier to distinguish between healthy individuals and those with Parkinson's based on vocal feature extractions, prioritizing model interpretability and the minimization of False Negatives (missed diagnoses).

---

##  Dataset

The dataset used is the **UCI Oxford Parkinson's Disease dataset** (`parkinsons.csv`). 
> **https://archive.ics.uci.edu/dataset/174/parkinsons**
> 
It contains a range of biomedical voice measurements from 31 people, 23 of whom have Parkinson's disease. Each column is a particular voice measure, and each row corresponds to one of 195 voice recordings from these individuals.

> **Important Data Handling:** The script actively extracts Patient IDs from the dataset to ensure strict group boundaries during training and testing, preventing data leakage.



---

##  Methodology & Key Features

1. **Patient-Grouped Splitting (`GroupShuffleSplit`)**:
Standard random splitting causes data leakage when multiple recordings exist per patient. This pipeline groups data by Patient ID, ensuring a patient's recordings exist *entirely* in the training set or *entirely* in the test set.
2. **Hyperparameter Tuning for Recall**:
Using `GridSearchCV` and `StratifiedGroupKFold`, the Random Forest is optimized specifically for **Recall (Sensitivity)** to ensure the model catches as many true positive Parkinson's cases as possible.
3. **Custom Clinical Thresholding**:
By default, models classify based on a 50% probability threshold. In clinical screening, missing a diagnosis is more harmful than a false alarm. The script evaluates a standard 50% threshold and dynamically shifts to a **30% threshold**, aggressively reducing false negatives.
4. **Model Clarity (SHAP)**:
Integrated `shap` to generate summary plots, allowing clinicians to see exactly which acoustic features drive the model's predictions.

---

##  Results & Clinical Relevance

Based on the model execution, the pipeline outputs several critical evaluation metrics:

* **Top Diagnostic Features**: The model identifies features like `PPE`, `spread1`, and `spread2` as the most crucial indicators of the disease.
* **Optimized Parameters**: GridSearch determined the best Random Forest configuration (e.g., `max_depth: None`, `min_samples_leaf: 1`, `n_estimators: 50`).
* **Clinical Threshold Shift**:
* At a **50% threshold**, the model achieves an overall baseline accuracy of ~69.7% with a standard confusion matrix (see `output part b.PNG`).
* At a **30% threshold**, the model achieves **100% recall** on the test set, successfully flagging all 31 Parkinson's recordings without a single False Negative, at the expense of higher False Positives (see `output part c.PNG`).


* **ROC Analysis**: Visualized via the Receiver Operating Characteristic curve (see `output part d.PNG`).
* **SHAP Interpretability**: The SHAP summary plot (see `output part e.PNG`) reveals that lower fundamental frequencies (`MDVP:Fo(Hz)`) and higher values in nonlinear measures of fundamental frequency variation (`spread1`) heavily push the model toward a Parkinson's prediction.

---

##  Requirements

To run this script, you will need Python 3.7+ and the following libraries:

```bash
pip install pandas numpy matplotlib scikit-learn shap joblib

```

---

##  Usage

1. Clone the repository and ensure `parkinsons.csv` is in the root directory.
2. Run the main analysis script:
```bash
python main.py

```


3. The script will output terminal reports, display interactive `matplotlib` charts (Confusion Matrices, ROC Curve, and SHAP Summary), and finally save the trained model as `clinical_rf_pipeline_parkinsons.pkl` for future inference.
