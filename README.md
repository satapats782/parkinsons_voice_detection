# Parkinson’s Disease Voice Detection

This project applies machine learning to detect Parkinson’s disease** using the [UCI Parkinson’s Voice Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons).  
The dataset consists of biomedical voice measurements from individuals with and without Parkinson’s disease.  
Our goal is to benchmark multiple imputation strategies and classifiers to build a robust detection pipeline.


## Objective
- **Task:** Binary classification (`status` → 1 = Parkinson’s, 0 = Healthy)
- **Features:** 22 biomedical voice measures (jitter, shimmer, HNR, PPE, spread1/2, etc.)
- **Target:** `status`


## ⚡ Features
- Handles missing values with multiple imputers:
  - Mean, Median, KNN, Iterative
- Benchmarks multiple classifiers:
  - Logistic Regression, Random Forest, Gradient Boosting, SVM
- Evaluation with **Repeated Stratified K-Fold CV**
- Reports **Accuracy, F1-score, and ROC-AUC**
- Saves results in `results_parkinsons.csv`

---

## Installation

### Requirements
- Python 3.8+
- pandas  
- numpy  
- scikit-learn  
- ydata-profiling (optional, for dataset profiling)  
- matplotlib (optional, for ROC plots)

Install dependencies with:
```bash
pip install -r requirements.txt

##Usage
Train & Evaluate Models

python3 parkinsons_classifier.py --data parkinsons_data.csv --target status
This will:

Run CV with multiple imputers + classifiers

Print results to the console

Save leaderboard to results_parkinsons.csv
