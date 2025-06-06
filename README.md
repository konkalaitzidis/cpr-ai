# CPR-AI: Clinical Pathway Recommender AI

This project has been developed as part of the Applied Artificial Intelligence (AI) in Healthcare course at Karolinska Institutet. The course provides fundamental knowledge about AI and its applications in healthcare, covering topics such as medical image analysis, data analytics, and decision support systems. More information about the course can be found [here](https://education.ki.se/course-syllabus/2QA338).

This project implements a machine learning-based system for recommending clinical procedures based on patient demographics and conditions. The system uses a Random Forest model trained on clinical data to predict the most appropriate next procedure for a patient.

## Features

- Web-based interface using Streamlit
- Machine learning model for procedure recommendations
- Support for multiple patient conditions
- Easy-to-use input form for patient demographics and conditions

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/konkalaitzidis/cpr-ai.git
cd cpr-ai
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model (if needed):
```bash
python src/train.py
```

2. Run the Streamlit app:
```bash
streamlit run src/app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Project Structure

```
cpr-ai/
├── data/               # Data directory
├── models/            # Trained models and related files
├── src/               # Source code
│   ├── app.py        # Streamlit application
│   ├── train.py      # Model training script
│   └── utils.py      # Utility functions
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Model Details

The system uses a Random Forest classifier trained on patient demographics and conditions to predict the next clinical procedure. The model takes into account:
- Patient age
- Gender
- Encounter type
- Existing conditions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
