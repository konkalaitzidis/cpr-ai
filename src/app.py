import streamlit as st
import pandas as pd
import joblib

# Load model and labels
model = joblib.load("models/rf_model.joblib")
label_df = pd.read_csv("models/labels.csv")
label_map = {i: label for i, label in enumerate(label_df['TARGET_PROCEDURE'].unique())}

st.title("🩺 Clinical Pathway Recommender")
st.write("Enter patient info to get the next recommended clinical procedure.")

# Input fields
age = st.slider("Age", 0, 100, 30)
gender = st.radio("Gender", ("Male", "Female"))
encounter_type = st.selectbox("Encounter Type", ["ambulatory", "emergency", "inpatient", "wellness", "urgentcare"])

# Map inputs
is_male = 1 if gender == "Male" else 0
encounter_type_map = {
    "ambulatory": 0,
    "emergency": 1,
    "inpatient": 2,
    "urgentcare": 3,
    "wellness": 4
}
enc_type_code = encounter_type_map.get(encounter_type, 0)

# Predict
if st.button("Get Recommendation"):
    X = pd.DataFrame([[age, is_male, enc_type_code]], columns=["AGE", "IS_MALE", "ENCOUNTER_TYPE"])
    pred = model.predict(X)[0]
    recommended_procedure = label_map.get(pred, "Unknown")

    st.success(f"✅ Recommended Procedure: **{recommended_procedure}**")
