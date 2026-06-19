import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Load the data
# This script lives in the "train" folder, and the CSV lives in a
# separate "data" folder one level up. Using __file__ to build the
# path means this works no matter which folder you run the script
# from (instead of relying on the current working directory).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "parkinsons_disease_data.csv")
df = pd.read_csv(data_path)

# 2. Separate features and target
# "PatientID" is just a row number, not a real medical feature.
# "DoctorInCharge" has the exact same value in every row (just a
# confidentiality placeholder), so it carries zero information either.
# Both must be dropped, otherwise the model wastes feature slots on
# data that means nothing.
X = df.drop(columns=["PatientID", "DoctorInCharge", "Diagnosis"])
y = df["Diagnosis"]

feature_columns = list(X.columns)
print("Number of real features used:", len(feature_columns))

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train the model
# A scaler is added here because features have very different ranges
# (e.g. UPDRS score vs SystolicBP vs BMI). Without scaling,
# LogisticRegression throws a "failed to converge" warning and
# performs slightly worse.
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)

# 5. Check accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Test accuracy:", acc)
print("\nClassification report:\n", classification_report(y_test, pred))

# 6. Save the model + column order
# Column order is saved so the prediction form later sends values
# in the exact same order the model was trained on.
models_dir = os.path.join(BASE_DIR, "..", "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, os.path.join(models_dir, "parkinsons_model.pkl"))
joblib.dump(feature_columns, os.path.join(models_dir, "parkinsons_columns.pkl"))

print("Parkinson's model saved successfully!")