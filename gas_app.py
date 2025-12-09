import streamlit as st
import requests

# FastAPI endpoint URL
FASTAPI_URL = "http://127.0.0.1:8000/predict_gas"

st.title("Gas Type Prediction (XGBoost & CatBoost)")

st.write("Pilih model dan cara input nilai sensor (UI).")

# ===================== PILIH MODEL =====================
model_choice = st.selectbox(
    "Pilih model:",
    ["XGBoost", "CatBoost"]   
)

# ===================== TAB UNTUK 2 UI =====================
tab1, tab2 = st.tabs(["Slider UI (original)", "Number Input UI"])

# ------- TAB 1: SLIDER UI (VERSI GITHUB) -------
with tab1:
    st.subheader("Input dengan Slider")

    MQ2 = st.slider("MQ2", min_value=0, max_value=1000, value=0)
    MQ3 = st.slider("MQ3", min_value=0, max_value=1000, value=0)
    MQ5 = st.slider("MQ5", min_value=0, max_value=1000, value=0)
    MQ6 = st.slider("MQ6", min_value=0, max_value=1000, value=0)
    MQ7 = st.slider("MQ7", min_value=0, max_value=1000, value=500)
    MQ8 = st.slider("MQ8", min_value=0, max_value=1000, value=0)
    MQ135 = st.slider("MQ135", min_value=0, max_value=1000, value=500)

    if st.button("Predict (Slider UI)", key="btn_slider"):
        data = {
            "MQ2": MQ2,
            "MQ3": MQ3,
            "MQ5": MQ5,
            "MQ6": MQ6,
            "MQ7": MQ7,
            "MQ8": MQ8,
            "MQ135": MQ135,
            # <- kirim "xgb" kalau pilih XGBoost, kalau tidak "cat"
            "model_name": "xgb" if model_choice == "XGBoost" else "cat"
        }

        try:
            response = requests.post(FASTAPI_URL, json=data)
            if response.status_code == 200:
                predictions = response.json()
                st.success(f"Model digunakan: {predictions['model']}")
                st.write(f"**Prediksi jenis gas:** `{predictions['prediction']}`")
            else:
                st.error(f"Error dari API: {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error("Error menghubungi FastAPI.")
            st.text(str(e))

# ------- TAB 2: NUMBER INPUT UI (VERSI KITA) -------
with tab2:
    st.subheader("Input dengan Number Input")

    col1, col2 = st.columns(2)

    with col1:
        mq2 = st.number_input("MQ2", min_value=0, max_value=1000, value=600)
        mq3 = st.number_input("MQ3", min_value=0, max_value=1000, value=400)
        mq5 = st.number_input("MQ5", min_value=0, max_value=1000, value=350)
        mq6 = st.number_input("MQ6", min_value=0, max_value=1000, value=380)

    with col2:
        mq7 = st.number_input("MQ7", min_value=0, max_value=1000, value=550)
        mq8 = st.number_input("MQ8", min_value=0, max_value=1000, value=600)
        mq135 = st.number_input("MQ135", min_value=0, max_value=1000, value=450)

    if st.button("Predict (Number Input UI)", key="btn_number"):
        data = {
            "MQ2": mq2,
            "MQ3": mq3,
            "MQ5": mq5,
            "MQ6": mq6,
            "MQ7": mq7,
            "MQ8": mq8,
            "MQ135": mq135,
            # lagi-lagi: "xgb" kalau XGBoost, else "cat"
            "model_name": "xgb" if model_choice == "XGBoost" else "cat"
        }

        try:
            response = requests.post(FASTAPI_URL, json=data)
            if response.status_code == 200:
                predictions = response.json()
                st.success(f"Model digunakan: {predictions['model']}")
                st.write(f"**Prediksi jenis gas:** `{predictions['prediction']}`")
            else:
                st.error(f"Error dari API: {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error("Error menghubungi FastAPI.")
            st.text(str(e))
