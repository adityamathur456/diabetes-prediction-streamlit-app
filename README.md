# Diabetes Prediction Streamlit App

An end-to-end Machine Learning web application built using Streamlit to predict whether a person is likely to have diabetes based on key medical parameters. The project demonstrates the complete ML workflow, including data preprocessing, model training, evaluation, explainability, and deployment.

---

## Project Overview

This application uses a **Support Vector Machine (SVM)** classifier trained on a diabetes dataset to make predictions from user-provided health inputs. The app provides interactive visualizations, SHAP-based model explainability, and generates downloadable PDF reports for predictions.

---

## Features

- Interactive Streamlit web interface  
- Diabetes prediction using SVM classifier  
- Automatic data preprocessing and scaling  
- Exploratory data analysis visualizations  
- SHAP-based model explainability (feature importance)  
- Downloadable PDF prediction report  
- Clean modular project structure  

---

## Tech Stack

- **Programming Language:** Python  
- **Web Framework:** Streamlit  
- **Machine Learning:** Scikit-learn (SVM)  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Model Explainability:** SHAP  
- **Report Generation:** ReportLab  

---

## Project Structure

```text
diabetes-prediction-streamlit-app/
│
├── app/
│   └── app.py                     # Streamlit application
│
├── dataset/
│   ├── graph-representation-of-dataset/
│   │   ├── correlation_heatmap.png
│   │   └── outcome_distribution.png
│   └── diabetes.csv               # Dataset
│
├── model/
│   ├── artifacts/                 # Trained model files
│   ├── data_preprocessing.py      # Data preprocessing logic
│   └── diabetes_model.py          # Model training code
│
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
└── .gitignore
```
---
## Installation Setup
```- git clone https://github.com/adityamathur456/diabetes-prediction-streamlit-app.git cd diabetes-prediction-streamlit-app ```

## Install dependencies
```- pip install -r requirements.txt```

## Run the Application
 ```- streamlit run app/app.py```
