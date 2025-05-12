import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("synthetic_clinical_journey.csv")

# Encode categorical variables
le_gender = LabelEncoder()
df['gender_enc'] = le_gender.fit_transform(df['gender'])

le_symptoms = LabelEncoder()
df['symptoms_enc'] = le_symptoms.fit_transform(df['symptoms'])

le_action = LabelEncoder()
df['action_enc'] = le_action.fit_transform(df['clinical_action'])

# Define features and target
features = ['age', 'gender_enc', 'heart_rate', 'systolic_bp', 'respiratory_rate', 'symptoms_enc', 'step']
X = df[features]
y = df['action_enc']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le_action.classes_))

# Save model + encoders (optional, for use in simulation later)
import joblib
joblib.dump(model, "rf_clinical_model.pkl")
joblib.dump(le_action, "action_encoder.pkl")
joblib.dump(le_symptoms, "symptoms_encoder.pkl")
joblib.dump(le_gender, "gender_encoder.pkl")
