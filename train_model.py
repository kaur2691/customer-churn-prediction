import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# -------------------------------------------
# LOAD DATA
# -------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# -------------------------------------------
# ENCODE CATEGORICAL COLUMNS
# -------------------------------------------
encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -------------------------------------------
# TRAIN MODEL
# -------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

model = RandomForestClassifier()
model.fit(X, y)

# -------------------------------------------
# FEATURE IMPORTANCE (Permutation Importance)
# -------------------------------------------
print("Calculating feature importance...")

result = permutation_importance(
    model,
    X,
    y,
    n_repeats=10,
    random_state=42
)

importance = result.importances_mean

plt.figure(figsize=(10, 6))
plt.barh(X.columns, importance)
plt.xlabel("Importance Score")
plt.title("Feature Importance (Permutation Importance)")
plt.tight_layout()
plt.savefig("feature_importance.png")

print("Saved: feature_importance.png")

# -------------------------------------------
# SAVE MODEL + ENCODERS
# -------------------------------------------
pickle.dump(model, open("ccp_rf_model.pkl", "wb"))
pickle.dump(encoders, open("ccp_encoders.pkl", "wb"))

print("Model training complete!")
print("Files saved:")
print("- ccp_rf_model.pkl")
print("- ccp_encoders.pkl")
print("- feature_importance.png")
