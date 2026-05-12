# Water Quality Prediction using Machine Learning

A Flask-based Machine Learning web application to predict whether water is potable or not using water quality parameters.

## Features
- Predict water quality
- Random Forest Classifier model
- Flask web interface
- Data preprocessing
- Imbalanced data handling
- Prediction results display

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML/CSS

## Project Structure

water_qual_prediction/
│── app.py  
│── main.py  
│── model_rfc.pkl  
│── requirements.txt  
│── README.md  
│── water_potability.csv  
│  
├── templates/  
│   ├── index.html  
│   └── result.html  
│  
├── src/  
│   ├── data_preprocessing.py  
│   ├── imbalance.py  
│   └── model.py  
│  
├── processed_data/  
└── report/

## Installation

Clone repository:

```bash
git clone https://github.com/Shiny12345shiny/Water-quality-prediction-project.git
cd Water-quality-prediction-project
