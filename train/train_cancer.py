import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# =========================
# 1. SAFE FILE PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "cancer.csv")

df = pd.read_csv(file_path)

# =========================
# 2. CLEAN DATA
# =========================
# Drop columns that are NOT real features:
# - "id" is just a row identifier, not medical data
# - "Unnamed: 32" is an empty column caused by a trailing comma
#   in the original CSV (it's all NaN)
# Using errors="ignore" so this still works if a column is already missing.
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Drop any other fully-empty columns just in case
df = df.dropna(axis=1, how="all")

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"].map({'M': 1, 'B': 0})

feature_columns = list(X.columns)
print("Number of real features used:", len(feature_columns))
print(feature_columns)

# =========================
# 3. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 4. SCALING (ONLY FIT ON TRAIN)
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5. MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =========================
# 6. PREDICTION + ACCURACY
# =========================
pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)

print("\nTest set accuracy:", acc)
print("\nClassification report:\n", classification_report(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print("\n5-fold cross-validation scores:", cv_scores)
print("Average CV accuracy:", cv_scores.mean())

# =========================
# 7. SAVE MODEL + SCALER + COLUMN ORDER
# =========================
models_path = os.path.join(BASE_DIR, "..", "models")
os.makedirs(models_path, exist_ok=True)

joblib.dump(model, os.path.join(models_path, "cancer_model.pkl"))
joblib.dump(scaler, os.path.join(models_path, "cancer_scaler.pkl"))
joblib.dump(feature_columns, os.path.join(models_path, "cancer_columns.pkl"))

print("\nModel, scaler, and column order saved successfully!")