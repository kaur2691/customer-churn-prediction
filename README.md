# Customer Churn Prediction

This project predicts whether a customer will churn using machine learning. It includes data preprocessing, model training, and a Streamlit app for easy predictions.

## Features
- Exploratory Data Analysis with visualizations
- RandomForest / CatBoost / XGBoost model for prediction
- SHAP explanations for feature importance
- Streamlit app deployment for interactive use

## How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/customer_churn_project.git
   cd customer_churn_project

2. Install dependencies:

  pip install -r requirements.txt


3.Train the model (optional if model is already saved):

  python model.py


4.Run Streamlit app:

  streamlit run app.py


---

## **5️⃣ model.py** (Train & Save Model)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("data/customer_data.csv")

# Preprocess dataset (adjust as per your data)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
with open("saved_models/churn_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved!")
