from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# ======= Siapkan label encoder (biar prediksi balik ke nama gas) =======
df = pd.read_csv("data/raw/Gas_Sensors_Measurements.csv")
le = LabelEncoder()
le.fit(df["Gas"])   # ['Mixture', 'NoGas', 'Perfume', 'Smoke'] -> 0,1,2,3

# ======= Load models (.pkl di root folder) =======
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))   # ganti RF -> XGB
cat_model = pickle.load(open("cat_model.pkl", "rb"))


# ======= Schema input =======
class SensorData(BaseModel):
    MQ2: float
    MQ3: float
    MQ5: float
    MQ6: float
    MQ7: float
    MQ8: float
    MQ135: float
    # sekarang default "xgb" (bukan "rf")
    model_name: str = "xgb"   # "xgb" atau "cat"


@app.get("/")
def root():
    return {"message": "Simple Gas API is running"}


# ======= Endpoint prediksi =======
@app.post("/predict_gas")
async def predict_gas(data: SensorData):

    input_data = np.array([[ 
        data.MQ2,
        data.MQ3,
        data.MQ5,
        data.MQ6,
        data.MQ7,
        data.MQ8,
        data.MQ135
    ]])

    # pilih model
    if data.model_name.lower() == "cat":
        raw_pred = cat_model.predict(input_data)[0]
        used_model = "CatBoost"
    else:
        raw_pred = xgb_model.predict(input_data)[0]
        used_model = "XGBoost"

    # raw_pred = angka (0–3) -> balik ke nama gas
    gas_label = le.inverse_transform([int(raw_pred)])[0]

    return {
        "model": used_model,
        "prediction": gas_label
    }
