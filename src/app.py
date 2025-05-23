import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("models/rf_model.joblib")

# Load label mapping for procedures
labels_df = pd.read_csv("models/labels.csv")
procedure_decoder = dict(zip(labels_df["label"], labels_df["procedure"]))

# Load saved features used during training
with open("models/features.txt") as f:
    feature_list = [line.strip() for line in f.readlines()]

# For demo, hardcode encounter_type encoding mapping (should match training LabelEncoder)
encounter_type_encoder = {
    0: "ambulatory",
    1: "emergency",
    2: "inpatient",
    3: "other",
    4: "virtual"
}

encounter_type_decoder = {v: k for k, v in encounter_type_encoder.items()}

# Top conditions (must match train.py top_conditions order)
top_conditions = [
    "Asthma",
    "Back Pain",
    "Coronary Artery Disease",
    "Depressive Disorder",
    "GERD",
    "Hypertension",
    "Hyperlipidemia",
    "Osteoarthritis",
    "Type 2 Diabetes Mellitus",
    "Urinary Tract Infection"
]

st.title("Clinical Pathway Recommendation Demo")

age = st.number_input("Patient Age", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", ["M", "F"])
encounter_type = st.selectbox("Encounter Type", list(encounter_type_decoder.keys()))

st.header("Patient Conditions")
condition_inputs = {}
for cond in top_conditions:
    condition_inputs[cond] = st.checkbox(cond)

# Button to predict
if st.button("Recommend Next Clinical Procedure"):
    # Prepare input dictionary with all features set to 0
    input_dict = {feat: 0 for feat in feature_list}

    input_dict["AGE"] = age
    input_dict["IS_MALE"] = 1 if gender == "M" else 0
    input_dict["ENCOUNTER_TYPE"] = encounter_type_decoder[encounter_type]

    # Set condition features
    for cond in top_conditions:
        feature_name = f"has_{cond.replace(' ', '_').lower()}"
        if feature_name in input_dict:
            input_dict[feature_name] = int(condition_inputs[cond])

    # Create DataFrame in correct column order
    input_df = pd.DataFrame([input_dict], columns=feature_list)

    # Predict procedure label
    pred_label = model.predict(input_df)[0]
    pred_procedure = procedure_decoder.get(pred_label, "Unknown Procedure")

    st.success(f"Recommended next clinical procedure: **{pred_procedure}**")
