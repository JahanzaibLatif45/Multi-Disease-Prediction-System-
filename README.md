# 🏥 Multi Disease Prediction System

A machine learning web application built with **Streamlit** that predicts the likelihood of **five diseases** — Diabetes, Heart Disease, Breast Cancer, Parkinson's Disease, and Kidney Disease — based on patient medical data.

🔗 **Live App:** https://multi-disease-prediction-system-5.streamlit.app/
---

## 📌 Features

- **5 Disease Predictions** — each with its own trained ML model
- **Probability Bar Chart** — shows confidence of each prediction visually
- **Data Visualization** — distribution graphs for each dataset
- **Model Accuracy & Feature Importance** — shows which features matter most for each model
- **Input Validation** — form values are constrained to realistic medical ranges from the actual dataset
- **Medical Disclaimer** — clearly marked as an educational AI tool, not a clinical diagnosis system

---

## 🧠 Diseases & Models

| Disease | Algorithm | Dataset | Test Accuracy |
|---|---|---|---|
| Diabetes | Random Forest | Pima Indians Diabetes (Kaggle) | ~78% |
| Heart Disease | Random Forest | UCI Heart Disease (Kaggle) | ~85% |
| Breast Cancer | Logistic Regression (Pipeline) | Wisconsin Breast Cancer (Kaggle) | ~96% |
| Parkinson's Disease | Logistic Regression (Pipeline) | Parkinson's Disease Data (Kaggle) | ~81% |
| Kidney Disease | Random Forest | Kidney Disease Dataset (Kaggle) | ~80% |

> ⚠️ Accuracy numbers reflect test set performance. The Kidney Disease model has limited ability to distinguish between risk categories due to weak feature-to-label correlation in the dataset — its predictions should be treated with extra caution.

---

## 🗂️ Project Structure

```
Multi-Disease-Prediction-System/
│
├── app.py                      # Main Streamlit application
│
├── train_diabetes.py           # Train diabetes model
├── train_heart.py              # Train heart disease model
├── train_cancer.py             # Train breast cancer model
├── train_parkinsons.py         # Train Parkinson's model
├── train_kidney.py             # Train kidney disease model
│
├── data/                       # Datasets (CSV files)
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── cancer.csv
│   ├── parkinsons_disease_data.csv
│   └── kidney_disease_dataset.csv
│
├── models/                     # Saved trained models (generated after running train scripts)
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── diabetes_columns.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── heart_columns.pkl
│   ├── cancer_model.pkl
│   ├── cancer_scaler.pkl
│   ├── cancer_columns.pkl
│   ├── parkinsons_model.pkl
│   ├── parkinsons_columns.pkl
│   ├── kidney_model.pkl
│   └── kidney_columns.pkl
│
└── requirements.txt            # Python dependencies
```

---

## ⚙️ How To Run Locally

### Step 1 — Clone the repository

```bash
git clone https://github.com/JahanzaibLatif45/Multi-Disease-Prediction-System.git
cd Multi-Disease-Prediction-System
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Train all models

Run each training script once. This generates the `.pkl` model files inside the `models/` folder.

```bash
python train_diabetes.py
python train_heart.py
python train_cancer.py
python train_parkinsons.py
python train_kidney.py
```

### Step 4 — Run the app

```bash
streamlit run app.py
```

---

## 📦 Dependencies

- Python 3.8+
- streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

---

## 📁 Datasets Used

| Disease | Dataset | Source |
|---|---|---|
| Diabetes | Pima Indians Diabetes Dataset |  
| Heart Disease | Heart Disease Dataset |  |
| Breast Cancer | Wisconsin Breast Cancer Dataset | 
| Parkinson's | Parkinson's Disease Dataset | 
| Kidney Disease | Kidney Disease Dataset |  

---



## ⚠️ Disclaimer

This application is built for **educational and demonstration purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns.

---

## 👤 Author

**Jahanzaib Latif**
BS Computer Science — Bahauddin Zakariya University

- LinkedIn: [jahanzaib-latif](https://www.linkedin.com/in/jahanzaib-latif-063907373)
- GitHub: [JahanzaibLatif45](https://github.com/JahanzaibLatif45)
