import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# =========================
# 1. SAFE PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "heart.csv")

df = pd.read_csv(file_path)

# =========================
# 2. CLEAN DATA
# =========================
df = df.dropna()  # remove missing values

# =========================
# 3. FEATURES + TARGET
# =========================
X = df.drop("target", axis=1)
y = df["target"]

# IMPORTANT: save the exact column order used for training.
# When you predict new data later, you must build the input
# DataFrame with these same columns in this same order,
# otherwise RandomForest will misinterpret which value belongs
# to which feature.
feature_columns = list(X.columns)

# =========================
# 4. SPLIT (stratify keeps class balance same in train/test)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 5. SCALING
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 6. MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =========================
# 7. EVALUATE (test set + cross-validation)
# =========================
pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)

print("Test set accuracy:", acc)
print("\nClassification report:\n", classification_report(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# Cross-validation tells you the REAL, honest accuracy.
# If this number is much lower than your test accuracy,
# your model is overfitting.
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print("\n5-fold cross-validation scores:", cv_scores)
print("Average CV accuracy:", cv_scores.mean())

# =========================
# 8. SAVE MODEL + SCALER + COLUMN ORDER
# =========================
models_path = os.path.join(BASE_DIR, "..", "models")
os.makedirs(models_path, exist_ok=True)

joblib.dump(model, os.path.join(models_path, "heart_model.pkl"))
joblib.dump(scaler, os.path.join(models_path, "heart_scaler.pkl"))
joblib.dump(feature_columns, os.path.join(models_path, "heart_columns.pkl"))

print("\nModel, scaler, and column order saved successfully!")