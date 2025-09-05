import argparse
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
np.random.seed(42)

def build_imputers():
    return {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "knn": KNNImputer(n_neighbors=3),
        "iter": IterativeImputer(random_state=42),
    }

def build_models():
    return {
        "logreg": LogisticRegression(max_iter=2000, solver="liblinear"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
        "gbr": GradientBoostingClassifier(random_state=42),
        "svm": SVC(probability=True, kernel="rbf", random_state=42),
    }

def evaluate(X, y, imputer, model, splits=5, repeats=2):
    pipe = Pipeline([
        ("imputer", imputer),
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=42)
    scoring = {
        "ACC": make_scorer(accuracy_score),
        "F1": make_scorer(f1_score),
        "AUC": make_scorer(roc_auc_score, needs_proba=True),
    }
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return np.mean(scores["test_ACC"]), np.mean(scores["test_F1"]), np.mean(scores["test_AUC"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV file with data")
    parser.add_argument("--target", type=str, default="status", help="Target column")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y = df[args.target].values
    X = df.drop(columns=[args.target])
    if "name" in X.columns:
        X = X.drop(columns=["name"])  # drop non-numeric ID column

    imputers = build_imputers()
    models = build_models()

    results = []
    for imp_name, imputer in imputers.items():
        for model_name, model in models.items():
            acc, f1, auc = evaluate(X, y, imputer, model)
            results.append({
                "imputer": imp_name,
                "model": model_name,
                "ACC": acc,
                "F1": f1,
                "AUC": auc,
            })
            print(f"{imp_name:6s} + {model_name:6s} -> ACC={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")

    df_results = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
    out_csv = Path("results_parkinsons.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\n[INFO] Results saved to {out_csv}")

if __name__ == "__main__":
    main()
