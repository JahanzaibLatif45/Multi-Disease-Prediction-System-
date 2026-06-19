import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Load the data
# This script lives in the "train" folder, and the CSV lives in a
# separate "data" folder one level up. Using __file__ to build the
# path means this works no matter which folder you run the script
# from (instead of relying on the current working directory).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "kidney_disease_dataset.csv")
df = pd.read_csv(data_path)

# 2. Encode categorical (text) columns into numbers
# Most ML models can't work with text directly, so yes/no,
# normal/abnormal, present/not present type columns need to
# become 0/1 (or 0/1/2 for the 3-level "Physical activity level").
binary_maps = {
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

for col, mapping in binary_maps.items():
    df[col] = df[col].map(mapping)

# 3. Features + target
# IMPORTANT: this target is NOT simple Yes/No like the other diseases.
# It has 5 classes: No_Disease, Low_Risk, Moderate_Risk, High_Risk,
# Severe_Disease. This is a multi-class problem, and the classes are
# heavily imbalanced (most patients are No_Disease, very few are
# Severe_Disease).
X = df.drop(columns=["Target"])
y = df["Target"]

feature_columns = list(X.columns)
print("Number of features used:", len(feature_columns))

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train the model
# RandomForest is used here instead of LogisticRegression because:
#  - class_weight="balanced" helps the model pay attention to the
#    rare classes (Severe_Disease, High_Risk) instead of just always
#    predicting "No_Disease" to get a high accuracy score
#  - features have very different scales (urine specific gravity is
#    ~1.00-1.03, white blood cell count is in the thousands) and
#    RandomForest does not need that scaled the way LogisticRegression does
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# 6. Check accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Test accuracy:", acc)
print("\nClassification report:\n", classification_report(y_test, pred))

# 7. Save the model + column order
# Column order must be saved because this dataset has 41 feature
# columns, and the prediction form must send values in this exact order.
#
# compress=("xz", 9) shrinks the saved file a lot (roughly 32MB -> ~5MB
# for this model) with no change to the model itself or its accuracy —
# it's just a smaller, more compressed way of storing the exact same
# trained trees. The trade-off is that loading the file is a little
# slower, which is not noticeable for an app like this.
models_dir = os.path.join(BASE_DIR, "..", "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, os.path.join(models_dir, "kidney_model.pkl"), compress=("xz", 9))
joblib.dump(feature_columns, os.path.join(models_dir, "kidney_columns.pkl"))

print("Kidney model saved successfully!")