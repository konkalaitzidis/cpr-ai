# src/recommend.py

def recommend_next_step(diagnosis, age_group):
    if diagnosis == 'Chest pain':
        if age_group == 'Senior':
            return 'Chest X-ray'
        elif age_group == 'Adult':
            return 'ECG'
        else:
            return 'Consult pediatrician'
    
    elif diagnosis == 'Fever':
        if age_group == 'Child':
            return 'Throat swab'
        else:
            return 'CBC'
    
    elif diagnosis == 'Headache':
        return 'Neurology consult'
    
    else:
        return 'General physician assessment'
