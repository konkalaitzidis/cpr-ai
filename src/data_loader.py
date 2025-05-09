# src/data_loader.py

import pandas as pd
import random

def generate_patient_data(num_patients: int = 100) -> pd.DataFrame:
    genders = ['M', 'F']
    symptoms_list = ['chest pain', 'fever', 'fatigue', 'headache']
    vitals_list = ['BP:120/80', 'BP:140/90', 'Temp:37.5', 'HR:90', 'HR:102']
    diagnoses = ['angina', 'flu', 'migraine']
    next_steps = ['ECG', 'blood test', 'MRI', 'X-ray']

    data = []
    for i in range(num_patients):
        patient = {
            'patient_id': f'{i:03}',
            'age': random.randint(20, 90),
            'gender': random.choice(genders),
            'symptoms': random.choice(symptoms_list),
            'vitals': random.sample(vitals_list, 2),
            'diagnosis': random.choice(diagnoses),
            'next_step': random.choice(next_steps)
        }
        patient['vitals'] = '; '.join(patient['vitals'])
        data.append(patient)

    return pd.DataFrame(data)

def save_to_csv(df: pd.DataFrame, path: str = "data/simulated_patients.csv"):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = generate_patient_data(100)
    save_to_csv(df)
    print("✅ Patient data generated and saved.")
