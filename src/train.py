import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

os.makedirs("models", exist_ok=True)

# === Load Data ===
patients = pd.read_csv("data/synthea_sample/patients.csv")
encounters = pd.read_csv("data/synthea_sample/encounters.csv")
procedures = pd.read_csv("data/synthea_sample/procedures.csv")
conditions = pd.read_csv("data/synthea_sample/conditions.csv")

# === Merge Patient + Encounter ===
df = encounters.merge(patients, left_on="PATIENT", right_on="Id")
print("After first merge, columns:", df.columns.tolist())

df["AGE"] = 2020 - pd.to_datetime(df["BIRTHDATE"]).dt.year  # Approximate age
df["IS_MALE"] = df["GENDER"].apply(lambda x: 1 if x == "M" else 0)
df = df.rename(columns={"ENCOUNTERCLASS": "ENCOUNTER_TYPE"})

# === Use most recent procedure per encounter ===
latest_procedures = procedures.sort_values("START").drop_duplicates("ENCOUNTER", keep="last")
df = df.merge(latest_procedures[["ENCOUNTER", "DESCRIPTION"]], left_on="Id_x", right_on="ENCOUNTER", how="inner")
df = df.rename(columns={"DESCRIPTION_y": "PROCEDURE"})

# Filter: Keep only common procedures (appearing ≥ 50 times)
procedure_counts = df["PROCEDURE"].value_counts()
top_procedures = procedure_counts[procedure_counts >= 50].index
df = df[df["PROCEDURE"].isin(top_procedures)]

# === Extract most common diagnoses ===
conditions = conditions.rename(columns={"ENCOUNTER": "ENCOUNTER_ID", "DESCRIPTION": "CONDITION"})
condition_counts = conditions["CONDITION"].value_counts().nlargest(10)
top_conditions = list(condition_counts.index)

# Add binary columns for top 10 conditions per encounter
def extract_conditions(encounter_id):
    conds = conditions[conditions["ENCOUNTER_ID"] == encounter_id]["CONDITION"].tolist()
    return [1 if cond in conds else 0 for cond in top_conditions]

condition_features = df["Id_x"].apply(extract_conditions)
condition_df = pd.DataFrame(condition_features.tolist(), columns=[f"has_{c.replace(' ', '_').lower()}" for c in top_conditions])
df = pd.concat([df.reset_index(drop=True), condition_df], axis=1)

print("\nAfter condition features shape:", df.shape)
print("After condition features columns:", df.columns.tolist())

# === Select Features and Target ===
features = ["AGE", "IS_MALE", "ENCOUNTER_TYPE"] + list(condition_df.columns)
print("\nSelected features:", features)

df = df.dropna(subset=["PROCEDURE"])
df = df[df["ENCOUNTER_TYPE"].notna()]
df = df[df["PROCEDURE"].notna()]

print("\nAfter filtering shape:", df.shape)

# Encode categorical values
df["ENCOUNTER_TYPE"] = LabelEncoder().fit_transform(df["ENCOUNTER_TYPE"])
df["PROCEDURE_LABEL"] = LabelEncoder().fit_transform(df["PROCEDURE"])
procedure_encoder = dict(zip(df["PROCEDURE_LABEL"], df["PROCEDURE"]))

X = df[features]
y = df["PROCEDURE_LABEL"]

# === Save features list for later use in prediction ===
with open("models/features.txt", "w") as f:
    for feat in features:
        f.write(feat + "\n")

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# === Evaluation ===
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# === Save Model and Label Mapping ===
joblib.dump(clf, "models/rf_model.joblib")
pd.DataFrame(list(procedure_encoder.items()), columns=["label", "procedure"]).to_csv("models/labels.csv", index=False)
