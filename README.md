# 🔥 Forest Fire Burned Area Prediction

## 📌 Project Overview

This project predicts the burned area (in hectares) of forest fires in Portugal using machine learning models trained on the UCI Forest Fires dataset.

The project includes:

- Data preprocessing
- Log transformation of burned area
- Model training (Linear Regression, Random Forest, Gradient Boosting)
- Model evaluation (RMSE, MAE, R²)
- Feature importance analysis
- Gradio web application for prediction

---

## 📊 Dataset

Dataset: UCI Forest Fires Dataset  
Features include weather data, Fire Weather Index (FWI) components, and spatial grid coordinates.

---

## 🤖 Models Used

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Random Forest was selected for deployment.

---

## 📈 Evaluation Metrics

- RMSE
- MAE
- R² Score

Due to heavy skew in burned area distribution, log transformation was applied.

---

## 🖥 Web Application

The project includes a Gradio-based web application (`app.py`) that allows users to input:

- Grid coordinates
- Month and day
- FWI indices
- Weather parameters

And get predicted burned area instantly.

Run locally:

```bash
python app.py
