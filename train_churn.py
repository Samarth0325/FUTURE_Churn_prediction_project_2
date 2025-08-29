
"""
train_churn.py

A simple end-to-end churn prediction pipeline:
- Loads the sample CSV
- Preprocesses (categorical encoding, scaling)
- Trains Logistic Regression and Random Forest
- Evaluates on test set (classification report, ROC AUC)
- Saves the best model (joblib)

Run:
    python train_churn.py --data sample_churn_data.csv --out_dir outputs
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support
import joblib

def load_data(path):
    return pd.read_csv(path)

def build_pipeline(cat_features, num_features):
    # Impute numeric, scale; impute categorical and one-hot encode
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preproc = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features)
    ])

    return preproc

def train_and_evaluate(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    target = "churn"
    X = df.drop(columns=[target])
    y = df[target]

    # Identify feature types
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    preproc = build_pipeline(cat_features, num_features)

    # Logistic Regression pipeline
    pipe_lr = Pipeline([
        ("preproc", preproc),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
    ])

    # Random Forest pipeline
    pipe_rf = Pipeline([
        ("preproc", preproc),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
    ])

    # Fit both
    print("Training Logistic Regression...")
    pipe_lr.fit(X_train, y_train)
    print("Training Random Forest...")
    pipe_rf.fit(X_train, y_train)

    # Evaluate
    models = {"logistic": pipe_lr, "random_forest": pipe_rf}
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
        report = classification_report(y_test, y_pred, digits=4)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {"report": report, "auc": auc, "confusion_matrix": cm}
        print(f"== {name} ==")
        print(report)
        print("ROC AUC:", auc)
        print("Confusion matrix:\\n", cm)
        print("-"*40)

    # Choose best by AUC (fallback to accuracy-like if auc None)
    best_name = max(results.keys(), key=lambda k: (results[k]["auc"] if results[k]["auc"] is not None else 0))
    best_model = models[best_name]
    print("Best model:", best_name)

    # Save model and preprocessing
    joblib.dump(best_model, os.path.join(out_dir, "best_model.joblib"))
    print("Saved best model to", os.path.join(out_dir, "best_model.joblib"))

    # Save metrics
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        for name, r in results.items():
            f.write(f"Model: {name}\\n")
            f.write(f"AUC: {r['auc']}\\n")
            f.write("Classification report:\\n")
            f.write(r["report"] + "\\n")
            f.write("Confusion matrix:\\n")
            f.write(str(r["confusion_matrix"]) + "\\n")
            f.write("-"*40 + "\\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="sample_churn_data.csv")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    df = load_data(args.data)
    train_and_evaluate(df, args.out_dir)
