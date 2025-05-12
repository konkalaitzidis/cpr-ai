import numpy as np
import pandas as pd
import random

def simulate_patient_journey(patient_id, max_steps=4):
    age = np.random.randint(20, 90)
    gender = random.choice(['Male', 'Female'])
    heart_rate = np.random.randint(60, 120)
    systolic_bp = np.random.randint(90, 180)
    respiratory_rate = np.random.randint(12, 30)
    symptoms = random.choice(['chest pain', 'fever', 'headache', 'abdominal pain'])

    actions = []
    for step in range(max_steps):
        # Simulate new vitals based on previous ones
        heart_rate += np.random.randint(-5, 6)
        systolic_bp += np.random.randint(-10, 11)
        respiratory_rate += np.random.randint(-2, 3)

        # Determine clinical action based on rules
        if heart_rate > 110 or symptoms == 'chest pain':
            action = 'ECG'
        elif symptoms == 'fever':
            action = 'Blood Test'
        elif symptoms == 'abdominal pain':
            action = 'Ultrasound'
        elif symptoms == 'headache':
            action = 'CT Scan'
        else:
            action = 'Observation'

        actions.append({
            'patient_id': patient_id,
            'step': step + 1,
            'age': age,
            'gender': gender,
            'heart_rate': heart_rate,
            'systolic_bp': systolic_bp,
            'respiratory_rate': respiratory_rate,
            'symptoms': symptoms,
            'clinical_action': action
        })

        # Slightly shift symptom to simulate evolution
        if symptoms == 'fever' and step == 1:
            symptoms = 'headache'

    return actions

def generate_dataset(num_patients=1000, steps_per_patient=4):
    all_records = []
    for pid in range(num_patients):
        journey = simulate_patient_journey(pid, steps_per_patient)
        all_records.extend(journey)
    return pd.DataFrame(all_records)

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("/Users/konstantinoskalaitzidis/Desktop/ai@ki/cpr-ai/data/synthetic_clinical_journey.csv", index=False)
    print(df.head(10))
