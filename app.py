import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# SAFE LOADERS

def load_pickle(path):
    if not os.path.exists(path):
        st.error(f"❌ File not found: {path}")
        return None
    return joblib.load(path)

def load_data(path):
    if not os.path.exists(path):
        st.error(f"❌ Dataset not found: {path}")
        return None
    return pd.read_csv(path)



def build_inputs(df, columns):
    inputs = {}
    col1, col2 = st.columns(2)
    half = len(columns) // 2

    for i, col in enumerate(columns):
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        default_val = float(df[col].median())

        target_col = col1 if i < half else col2
        with target_col:
            inputs[col] = st.number_input(
                col, min_value=min_val, max_value=max_val, value=default_val
            )

    return inputs


# Kidney has several yes/no, normal/abnormal type columns that need to
# be shown as dropdowns (not number inputs) and converted to the same
# numbers used during training.
KIDNEY_CATEGORY_MAPS = {
    "Red blood cells in urine": {"normal": 0, "abnormal": 1},
    "Pus cells in urine": {"normal": 0, "abnormal": 1},
    "Pus cell clumps in urine": {"not present": 0, "present": 1},
    "Bacteria in urine": {"not present": 0, "present": 1},
    "Hypertension (yes/no)": {"no": 0, "yes": 1},
    "Diabetes mellitus (yes/no)": {"no": 0, "yes": 1},
    "Coronary artery disease (yes/no)": {"no": 0, "yes": 1},
    "Appetite (good/poor)": {"good": 0, "poor": 1},
    "Pedal edema (yes/no)": {"no": 0, "yes": 1},
    "Anemia (yes/no)": {"no": 0, "yes": 1},
    "Family history of chronic kidney disease": {"no": 0, "yes": 1},
    "Smoking status": {"no": 0, "yes": 1},
    "Urinary sediment microscopy results": {"normal": 0, "abnormal": 1},
    "Physical activity level": {"low": 0, "moderate": 1, "high": 2},
}


def build_kidney_inputs(df, columns):
    inputs = {}
    col1, col2 = st.columns(2)
    half = len(columns) // 2

    for i, col in enumerate(columns):
        target_col = col1 if i < half else col2

        with target_col:
            if col in KIDNEY_CATEGORY_MAPS:
                options = list(KIDNEY_CATEGORY_MAPS[col].keys())
                choice = st.selectbox(col, options)
                inputs[col] = KIDNEY_CATEGORY_MAPS[col][choice]
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                default_val = float(df[col].median())
                inputs[col] = st.number_input(
                    col, min_value=min_val, max_value=max_val, value=default_val
                )

    return inputs


# HELPER: show accuracy + feature importance for a model
# scaler can be None for models that either don't need scaling
# (RandomForest) or already do their own scaling internally
# (a Pipeline that contains StandardScaler, like the Parkinson's model).

def show_model_performance(model, scaler, df, columns, target_col, label_map=None):
    X = df[columns]
    y = df[target_col]

    if label_map:
        y = y.map(label_map)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test
    test_preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, test_preds)

    st.write(f"Accuracy on unseen test data: **{acc*100:.2f}%**")

    # Tree-based models (RandomForest) expose feature_importances_
    # directly. A Pipeline wrapping LogisticRegression exposes
    # coefficients through its final step instead.
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        final_step = list(model.named_steps.values())[-1]
        if hasattr(final_step, "coef_"):
            coef = final_step.coef_
            importances = abs(coef[0]) if coef.ndim > 1 else abs(coef)

    if importances is not None:
        importance_df = pd.DataFrame({
            "Feature": columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"], color="#3b82f6")
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Which features matter most for this model")
        st.pyplot(fig)


# LOAD MODELS, SCALERS, COLUMN ORDER

diabetes_model = load_pickle(os.path.join(BASE_DIR, "models/diabetes_model.pkl"))
diabetes_scaler = load_pickle(os.path.join(BASE_DIR, "models/diabetes_scaler.pkl"))
diabetes_columns = load_pickle(os.path.join(BASE_DIR, "models/diabetes_columns.pkl"))

heart_model = load_pickle(os.path.join(BASE_DIR, "models/heart_model.pkl"))
heart_scaler = load_pickle(os.path.join(BASE_DIR, "models/heart_scaler.pkl"))
heart_columns = load_pickle(os.path.join(BASE_DIR, "models/heart_columns.pkl"))

cancer_model = load_pickle(os.path.join(BASE_DIR, "models/cancer_model.pkl"))
cancer_scaler = load_pickle(os.path.join(BASE_DIR, "models/cancer_scaler.pkl"))
cancer_columns = load_pickle(os.path.join(BASE_DIR, "models/cancer_columns.pkl"))

# Parkinson's model is a Pipeline (StandardScaler + LogisticRegression)
# saved as a single object, so there is no separate scaler file to load.
parkinsons_model = load_pickle(os.path.join(BASE_DIR, "models/parkinsons_model.pkl"))
parkinsons_columns = load_pickle(os.path.join(BASE_DIR, "models/parkinsons_columns.pkl"))

# Kidney model is a plain RandomForest, no scaling used during training.
kidney_model = load_pickle(os.path.join(BASE_DIR, "models/kidney_model.pkl"))
kidney_columns = load_pickle(os.path.join(BASE_DIR, "models/kidney_columns.pkl"))

# UI HEADER

st.title("🧠 Multi Disease Prediction System")
st.warning("⚠ This is an AI prediction system. Please visit a medical consultant for real diagnosis.")

menu = st.sidebar.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Cancer", "Parkinson's Disease", "Kidney Disease"]
)


# DIABETES

if menu == "Diabetes":
    st.subheader("Diabetes Prediction")

    df = load_data(os.path.join(BASE_DIR, "data/diabetes.csv"))

    if df is not None and diabetes_model is not None and diabetes_scaler is not None and diabetes_columns is not None:

        st.dataframe(df.head())

        # In this dataset, 0 is used as a placeholder for missing
        # values in these columns (a real patient can't have 0
        # blood pressure or 0 BMI). We hide 0 from the input ranges
        # so the form never suggests an impossible value.
        df_for_form = df.copy()
        zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_as_missing:
            df_for_form[col] = df_for_form[col].replace(0, np.nan)

        st.info("👉 Please enter patient medical values")
        inputs = build_inputs(df_for_form, diabetes_columns)

        if st.button("Predict Diabetes"):
            input_df = pd.DataFrame([inputs])[diabetes_columns]
            input_scaled = diabetes_scaler.transform(input_df)

            result = diabetes_model.predict(input_scaled)
            prob = diabetes_model.predict_proba(input_scaled)[0][1]

            if result[0] == 1:
                st.error(f"⚠ Diabetic Risk Detected ({prob*100:.2f}%)")
            else:
                st.success(f"✔ Not Diabetic ({(1-prob)*100:.2f}%)")

            fig, ax = plt.subplots()
            ax.bar(["No Diabetes", "Diabetes"], [1-prob, prob], color=["#22c55e", "#ef4444"])
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            st.warning("⚠ AI reading only — consult doctor")

        with st.expander("📊 Model Accuracy & Feature Importance"):
            show_model_performance(diabetes_model, diabetes_scaler, df, diabetes_columns, "Outcome")


# HEART

elif menu == "Heart Disease":
    st.subheader("Heart Disease Prediction")

    df = load_data(os.path.join(BASE_DIR, "data/heart.csv"))

    if df is not None and heart_model is not None and heart_scaler is not None and heart_columns is not None:

        st.dataframe(df.head())

        st.info("👉 Enter correct medical values carefully")
        inputs = build_inputs(df, heart_columns)

        if st.button("Predict Heart Disease"):
            input_df = pd.DataFrame([inputs])[heart_columns]
            input_scaled = heart_scaler.transform(input_df)

            result = heart_model.predict(input_scaled)
            prob = heart_model.predict_proba(input_scaled)[0][1]

            if result[0] == 1:
                st.error(f"⚠ Heart Disease Risk ({prob*100:.2f}%)")
            else:
                st.success(f"✔ No Risk ({(1-prob)*100:.2f}%)")

            fig, ax = plt.subplots()
            ax.bar(["No Risk", "Heart Disease"], [1-prob, prob], color=["#22c55e", "#ef4444"])
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            st.warning("⚠ AI reading only — consult doctor")

        with st.expander("📊 Model Accuracy & Feature Importance"):
            show_model_performance(heart_model, heart_scaler, df, heart_columns, "target")


# CANCER

elif menu == "Cancer":
    st.subheader("Cancer Prediction")

    df = load_data(os.path.join(BASE_DIR, "data/cancer.csv"))

    if df is not None and cancer_model is not None and cancer_scaler is not None and cancer_columns is not None:

        st.dataframe(df.head())

        st.info("👉 Enter cell measurement values carefully")
        inputs = build_inputs(df, cancer_columns)

        if st.button("Predict Cancer"):
            input_df = pd.DataFrame([inputs])[cancer_columns]
            input_scaled = cancer_scaler.transform(input_df)

            result = cancer_model.predict(input_scaled)
            prob = cancer_model.predict_proba(input_scaled)[0][1]

            if result[0] == 1:
                st.error(f"⚠ Cancer Detected ({prob*100:.2f}%)")
            else:
                st.success(f"✔ No Cancer Detected ({(1-prob)*100:.2f}%)")

            fig, ax = plt.subplots()
            ax.bar(["No Cancer", "Cancer"], [1-prob, prob], color=["#22c55e", "#ef4444"])
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            st.warning("⚠ This is AI reading only — consult medical expert immediately")

        with st.expander("📊 Model Accuracy & Feature Importance"):
            show_model_performance(
                cancer_model, cancer_scaler, df, cancer_columns,
                "diagnosis", label_map={'M': 1, 'B': 0}
            )


# PARKINSON'S DISEASE

elif menu == "Parkinson's Disease":
    st.subheader("Parkinson's Disease Prediction")

    df = load_data(os.path.join(BASE_DIR, "data/parkinsons_disease_data.csv"))

    if df is not None and parkinsons_model is not None and parkinsons_columns is not None:

        st.dataframe(df.head())

        st.info("👉 Please enter patient medical values")
        inputs = build_inputs(df, parkinsons_columns)

        if st.button("Predict Parkinson's Disease"):
            input_df = pd.DataFrame([inputs])[parkinsons_columns]

            # No separate scaling step here — the saved model is a
            # Pipeline that already scales internally before predicting.
            result = parkinsons_model.predict(input_df)
            prob = parkinsons_model.predict_proba(input_df)[0][1]

            if result[0] == 1:
                st.error(f"⚠ Parkinson's Risk Detected ({prob*100:.2f}%)")
            else:
                st.success(f"✔ No Parkinson's Risk ({(1-prob)*100:.2f}%)")

            fig, ax = plt.subplots()
            ax.bar(["No Risk", "Parkinson's"], [1-prob, prob], color=["#22c55e", "#ef4444"])
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            st.warning("⚠ AI reading only — consult doctor")

        with st.expander("📊 Model Accuracy & Feature Importance"):
            show_model_performance(parkinsons_model, None, df, parkinsons_columns, "Diagnosis")


# KIDNEY DISEASE

elif menu == "Kidney Disease":
    st.subheader("Kidney Disease Risk Prediction")

    df = load_data(os.path.join(BASE_DIR, "data/kidney_disease_dataset.csv"))

    if df is not None and kidney_model is not None and kidney_columns is not None:

        st.dataframe(df.head())

        st.info(
            "ℹ️ Note: during testing, this model showed limited ability to "
            "tell the risk categories apart (the dataset's features did not "
            "correlate strongly with the labels). Treat its predictions, "
            "especially for the rarer risk levels, with extra caution."
        )

        st.info("👉 Please enter patient medical values")
        inputs = build_kidney_inputs(df, kidney_columns)

        if st.button("Predict Kidney Disease Risk"):
            input_df = pd.DataFrame([inputs])[kidney_columns]

            result = kidney_model.predict(input_df)[0]
            probs = kidney_model.predict_proba(input_df)[0]
            classes = kidney_model.classes_

            st.error(f"⚠ Predicted Risk Category: {result}")

            fig, ax = plt.subplots()
            ax.barh(classes, probs, color="#3b82f6")
            ax.set_xlabel("Probability")
            ax.set_title("Probability across all risk categories")
            st.pyplot(fig)

            st.warning("⚠ AI reading only — consult doctor")

        with st.expander("📊 Model Accuracy & Feature Importance"):
            # The raw dataset still has text categories (yes/no,
            # normal/abnormal), so they need the same encoding used
            # during training before accuracy can be calculated.
            df_encoded = df.copy()
            for col, mapping in KIDNEY_CATEGORY_MAPS.items():
                df_encoded[col] = df_encoded[col].map(mapping)

            show_model_performance(kidney_model, None, df_encoded, kidney_columns, "Target")