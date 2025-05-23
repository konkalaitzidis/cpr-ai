import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load data
patients = pd.read_csv("data/synthea_sample/patients.csv")
encounters = pd.read_csv("data/synthea_sample/encounters.csv")
conditions = pd.read_csv("data/synthea_sample/conditions.csv")
procedures = pd.read_csv("data/synthea_sample/procedures.csv")

print("Initial data shapes:")
print(f"Encounters: {encounters.shape}")
print(f"Patients: {patients.shape}")
print(f"Procedures: {procedures.shape}")

# Merge and preprocess
df = encounters.merge(patients, left_on='PATIENT', right_on='Id', suffixes=('_enc', '_pat'))
print("\nAfter first merge:")
print(f"Shape: {df.shape}")
print("Columns:", df.columns.tolist())

# Feature engineering (simplified)
df['AGE'] = 2020 - pd.to_datetime(df['BIRTHDATE']).dt.year
df['IS_MALE'] = (df['GENDER'] == 'M').astype(int)
df['ENCOUNTER_TYPE'] = df['ENCOUNTERCLASS'].astype('category').cat.codes

# Label: most recent procedure for the patient
recent_proc = procedures.sort_values(by='START').drop_duplicates(subset='PATIENT', keep='last')
print("\nRecent procedures:")
print(f"Shape: {recent_proc.shape}")
print("Columns:", recent_proc.columns.tolist())

df = df.merge(recent_proc[['PATIENT', 'DESCRIPTION']], on='PATIENT')
print("\nAfter second merge:")
print(f"Shape: {df.shape}")
print("Columns:", df.columns.tolist())

df = df.rename(columns={'DESCRIPTION_y': 'TARGET_PROCEDURE'})
print("\nAfter rename:")
print(f"Shape: {df.shape}")
print("Columns:", df.columns.tolist())

df['TARGET_PROCEDURE'] = df['TARGET_PROCEDURE'].astype('category')
df = df.dropna(subset=['TARGET_PROCEDURE'])

# Features and labels
X = df[['AGE', 'IS_MALE', 'ENCOUNTER_TYPE']]
y = df['TARGET_PROCEDURE'].cat.codes

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and classes
joblib.dump(clf, 'models/rf_model.joblib')
df[['TARGET_PROCEDURE']].drop_duplicates().to_csv('models/labels.csv', index=False)
