import json
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")

# =======================
# Load Dataset
# =======================
df = pd.read_csv("../data/processed/cleaned_data.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Split train-val-test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Data Split Done!")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# =======================
# XGBoost Hyperparameter Tuning
# =======================
xgb = XGBClassifier(
    eval_metric="mlogloss",
    random_state=42,
    tree_method="auto"
)

xgb_param_grid = {
    "n_estimators": [300, 500, 700],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 1, 5],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1, 2, 4],
    "reg_alpha": [0, 0.5, 1],
}

xgb_search = RandomizedSearchCV(
    xgb,
    xgb_param_grid,
    n_iter=25,
    scoring="f1_macro",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("\n🔍 Tuning XGBoost...")
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
print("XGBoost Best Params:", xgb_search.best_params_)

# Cross Validation Score
xgb_cv_score = cross_val_score(xgb_best, X_train, y_train, scoring="f1_macro", cv=5)
print("XGBoost CV F1 Macro:", xgb_cv_score.mean())

# Validation Performance
xgb_pred = xgb_best.predict(X_val)
print("\n📊 XGBoost Validation Results:")
print(classification_report(y_val, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, xgb_pred))


# =======================
# CatBoost Hyperparameter Tuning
# =======================
cat = CatBoostClassifier(
    verbose=False,
    random_state=42,
    eval_metric="F1"
)

cat_param_grid = {
    "iterations": [300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.1],
    "depth": [4, 6, 10],
    "l2_leaf_reg": [1, 3, 5],
    "bagging_temperature": [0.3, 1, 5],
}

cat_search = RandomizedSearchCV(
    cat,
    cat_param_grid,
    n_iter=20,
    scoring="f1_macro",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("\n🔍 Tuning CatBoost...")
cat_search.fit(X_train, y_train)
cat_best = cat_search.best_estimator_
print("CatBoost Best Params:", cat_search.best_params_)

# Cross Validation Score
cat_cv_score = cross_val_score(cat_best, X_train, y_train, scoring="f1_macro", cv=5)
print("CatBoost CV F1 Macro:", cat_cv_score.mean())

# Validation Performance
cat_pred = cat_best.predict(X_val)
print("\n📊 CatBoost Validation Results:")
print(classification_report(y_val, cat_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, cat_pred))


# =======================
# Compare & Select Best Model
# =======================
xgb_f1 = f1_score(y_val, xgb_pred, average="macro")
cat_f1 = f1_score(y_val, cat_pred, average="macro")

best_model = xgb_best if xgb_f1 >= cat_f1 else cat_best
best_name = "XGBoost" if xgb_f1 >= cat_f1 else "CatBoost"

print(f"\n🏆 Best Model Selected: {best_name}  (F1={max(xgb_f1, cat_f1):.4f})")

# =======================
# Save Model & Results
# =======================
pickle.dump(best_model, open("../models/best_model.pkl", "wb"))

results = {
    "best_model": best_name,
    "xgb_best_params": xgb_search.best_params_,
    "cat_best_params": cat_search.best_params_,
    "xgb_cv_f1": xgb_cv_score.mean(),
    "cat_cv_f1": cat_cv_score.mean(),
    "xgb_val_f1": xgb_f1,
    "cat_val_f1": cat_f1
}

with open("../models/tuning_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Model Saved Successfully!")
print("📁 Saved to '../models/best_model.pkl' and tuning_results.json")
