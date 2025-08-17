# Barcelona Air Quality Monitoring and Prediction

This project demonstrates how **machine learning** and **cloud-based deployment** can be applied to tackle a real-world problem: forecasting **urban air quality**.  

Using Barcelona’s open environmental data, I built a **prediction pipeline** that:  
- Processes raw time-series and weather data  
- Trains and evaluates multiple ML models (Random Forest, Gradient Boosting, Elastic Net, SVM)  
- Provides **daily air quality forecasts** (PM10, NO₂, etc.)  
- Generates **health recommendations** based on predicted pollution levels  
- Delivers results through a **Flask-powered web application** with a clean UI  

---

## 🎯 Why This Project Matters to Recruiters

This project highlights my ability to:  
- **Work with messy real-world data** (cleaning, feature engineering, handling missing values)  
- **Apply ML techniques** for forecasting and model evaluation (R², MSE, MAE)  
- **Build end-to-end solutions** — from data science in Jupyter notebooks to serving predictions via Flask  
- **Communicate results** effectively through a web app and presentation slides  
- **Think in terms of impact**, offering actionable insights for citizens and policymakers  

---

## 🌍 Project Overview

Air pollution poses serious health risks, yet real-time and predictive tools are limited.  
This project solves that by offering:

- **Real-time air quality monitoring** from Barcelona’s Open Data API  
- **Short-term prediction** of pollutants (e.g., NO2, PM10) using ML models  
- **User recommendations** (e.g., avoid outdoor activity on high-pollution days)  
- A **simple web interface** for tourists, delivery workers, and local citizens  

The project was developed as part of a **Cloud Computing final project** at ESADE (December 2024).

---

## 📂 Repository Structure

- `index.html` → Web interface template (Flask frontend)  
- `styles.css` → Styling for the interface (responsive design)  
- `model.py` → Python backend with ML model and Flask API routes (`/predict`)  
- `model.ipynb` → Jupyter notebook for model training, testing, and evaluation  
- `outputs.txt` → Example outputs and logs from model predictions  
- `presentation.pdf` → Project presentation with AWS architecture & technical details  

---

## ⚙️ How It Works

1. **Data Sources**  
   - Weather & Air Quality data from [Barcelona Open Data API](https://opendata-ajuntament.barcelona.cat)  
   - Pollutants monitored: PM10, NO2, O3, etc.  

2. **Machine Learning**  
   - Data preprocessing: remove missing values, add temporal features (day, month, etc.)  
   - Models tested: Random Forest, Gradient Boosting, SVM, Elastic Net  
   - Final model selected based on performance (R², MSE, MAE)  

3. **Web Application**  
   - Users input a date (day, month, year)  
   - Flask backend calls the trained ML model  
   - Prediction, health behavior, and pollution scale are shown dynamically on the webpage  

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+  
- Flask  
- scikit-learn  
- pandas, numpy, matplotlib  

Install requirements:
```bash
pip install flask scikit-learn pandas numpy matplotlib
