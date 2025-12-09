import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pickle

# Load data
df = pd.read_csv("data/raw/Gas_Sensors_Measurements.csv")

X = df[["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]]
y = df["Gas"]

# encode label string → angka
le = LabelEncoder()
y_encoded = le.fit_transform(y)   # 'Mixture','NoGas',.. → 0,1,2,3

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==== XGBOOST ====
import os, json

# ====== XGBOOST (dengan logging) ======
os.makedirs("xgboost_info", exist_ok=True)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

# simpan evaluasi train & test
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# akurasi
pred_xgb = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, pred_xgb)
print("XGBoost Accuracy:", acc_xgb)

# simpan metadata training
info = {
    "model": "XGBoost",
    "accuracy": acc_xgb,
    "parameters": xgb.get_params()
}

with open("xgboost_info/training_info.json", "w") as f:
    json.dump(info, f, indent=4)

# simpan history loss
results = xgb.evals_result()
with open("xgboost_info/eval_history.json", "w") as f:
    json.dump(results, f, indent=4)

print("XGBoost log disimpan di folder xgboost_info/")


# ==== CATBOOST ====
cat = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    verbose=0
)
cat.fit(X_train, y_train)
pred_cat = cat.predict(X_test)
print("CatBoost Accuracy:", accuracy_score(y_test, pred_cat))

# ==== SAVE BOTH MODELS ====
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

with open("cat_model.pkl", "wb") as f:
    pickle.dump(cat, f)

print("Model tersimpan: xgb_model.pkl & cat_model.pkl")

