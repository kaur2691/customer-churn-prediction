import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import pickle
import numpy as np
import os
import shap
import matplotlib.pyplot as plt


st.title("Customer Churn Prediction 📉")

# -------------------------
def load_lottie(url):
    from requests import get
    r = get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_url = load_lottie("https://assets10.lottiefiles.com/packages/lf20_yr6zz3wv.json")


# -------------------------
# SIDEBAR MENU
# -------------------------
with st.sidebar:

    option_main = option_menu(
        "CONTENTS",
        ["MAIN", "DATASET", "DESCRIPTIVE ANALYSIS", "VISUALISATIONS", "PREDICTION"],
        icons=["house", "database", "info-circle", "bar-chart", "robot"],
        menu_icon="list",
        default_index=0
    )


# -------------------------
# MAIN PAGE
# -------------------------
if option_main == "MAIN":
    with st.form("Front"):
     col1, col2 = st.columns(2)

    with col1:
        st_lottie(lottie_url, key="main_anim", height=350)

    with col2:
        st.title("WELCOME TO CHURN PREDICTION SYSTEM 📊")
        btn = st.form_submit_button("LEARN MORE")
        
    if btn:
            st.markdown("""

Understanding customer behavior is essential for retaining valuable clients.  
This platform helps businesses:

- 🔍 **Analyze customer usage patterns**  
- 🤖 **Predict whether a customer is likely to churn**  
- 📊 **Visualize churn trends and insights**  
- 💡 **Make data-driven retention decisions**  

Let’s explore your customer data with intelligence and accuracy! 🚀
""")


# -------------------------
# DATASET PAGE
# -------------------------
elif option_main == "DATASET":
    st.header("Dataset Overview 📁")

    uploaded_file = st.file_uploader("Upload your Customer Churn CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df  # 🔥 SAVE dataset in session state
        
        st.success("Dataset uploaded successfully!")
        st.write("### Preview:")
        st.dataframe(df.head())

        st.write("### Shape:", df.shape)
        st.write("### Columns:", list(df.columns))


# -------------------------
# DESCRIPTIVE ANALYSIS PAGE
# -------------------------
elif option_main == "DESCRIPTIVE ANALYSIS":
    st.header("Descriptive Analysis 📈")
    st.write("Summary statistics, missing values, and churn distribution will appear here.")

    if "df" not in st.session_state:
        st.error("⚠ Please upload a dataset first (in the DATASET section).")
    else:
        df = st.session_state["df"]

        st.subheader("Summary Statistics")
        st.dataframe(df.describe(include="all"))

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Churn Value Counts")
        if "Churn" in df.columns:
            st.bar_chart(df["Churn"].value_counts())
        else:
            st.warning("Column 'Churn' not found in dataset.")


# -------------------------
# VISUALISATIONS PAGE
# -------------------------
elif option_main == "VISUALISATIONS":
    st.header(" Data Visualizations 📊")
    st.write("Customer churn visual insights such as churn rate, demographics, and usage patterns.")
    st.info("Upload dataset in the DATASET section to view visualizations.")


    if "df" in st.session_state:
        df = st.session_state["df"]     

        st.bar_chart(df.select_dtypes(include=['number']))
        st.line_chart(df.select_dtypes(include=['number']))
    else:
        st.warning("Please upload a dataset from the sidebar.")

    if "Churn" in df.columns and "tenure" in df.columns:
        st.subheader("Churn vs Tenure")
        st.line_chart(df.groupby("tenure")["Churn"].count())
    else:
        st.warning("Required columns not found.")


# -------------------------
# PREDICTION PAGE
# -------------------------
elif option_main == "PREDICTION":
    st.header("🔮 Customer Churn Prediction (with training, gauge & SHAP)")

    st.write("Use the model to predict whether a customer will churn.")

    if "df" in st.session_state:
        df = st.session_state["df"]  

#model = pickle.load(open("ccp_rf_model.pkl", "rb"))
#encoders = pickle.load(open("ccp_encoders.pkl", "rb"))


    # ---------- helper: ensure columns are in good shape ----------
    st.info("Tip: Ensure the dataset has column `Churn` (0/1 or Yes/No).")

    # ---------- model / encoders file paths ----------
    MODEL_PATH = "ccp_rf_model.pkl"
    ENCODERS_PATH = "ccp_encoders.pkl"

    # ---------- utility functions ----------
    def save_model_and_encoders(model, encoders):
        pickle.dump(model, open(MODEL_PATH, "wb"))
        pickle.dump(encoders, open(ENCODERS_PATH, "wb"))

    def load_model_and_encoders():
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
            model = pickle.load(open(MODEL_PATH, "rb"))
            encoders = pickle.load(open(ENCODERS_PATH, "rb"))
            return model, encoders
        return None, None

    # ---------- TRAIN MODEL (optional) ----------
    st.subheader("Model Training (optional)")
    st.write("You can train a fresh RandomForest model from the uploaded dataset. This will overwrite any saved model files.")

    train_col1, train_col2 = st.columns([3,1])
    with train_col1:
        n_estimators = st.number_input("n_estimators", value=100, min_value=10, max_value=1000, step=10)
        max_depth = st.number_input("max_depth (use 0 for None)", value=0, min_value=0, max_value=50, step=1)
    with train_col2:
        train_btn = st.button("Train Model")

    if train_btn:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
        except Exception as e:
            st.error("Please install scikit-learn (`pip install scikit-learn`).")
            st.stop()

        # Basic cleaning: if Churn is Yes/No -> convert to 1/0
        df_train = df.copy()
        if df_train["Churn"].dtype == object:
            df_train["Churn"] = df_train["Churn"].map({"Yes":1, "No":0})

        # Fill or drop NA's simply (you can expand later)
        df_train = df_train.dropna(axis=0)  # easier safe baseline

        # Encode categoricals
        encoders = {}
        for col in df_train.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col].astype(str))
            encoders[col] = le

        X = df_train.drop("Churn", axis=1)
        y = df_train["Churn"]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=int(n_estimators),
                                       max_depth=(None if int(max_depth)==0 else int(max_depth)),
                                       random_state=42)
        with st.spinner("Training model..."):
            model.fit(x_train, y_train)

        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"Model trained — test accuracy: {acc:.3f}")
        st.text("Classification report:")
        st.text(classification_report(y_test, preds, digits=3))

        # Save model & encoders
        save_model_and_encoders(model, encoders)
        st.success("Model and encoders saved to disk (ccp_rf_model.pkl, ccp_encoders.pkl).")

    # ---------- LOAD model if present ----------
    model, encoders = load_model_and_encoders()
    if model is None:
        st.warning("No saved model found. You can train one above or place `ccp_rf_model.pkl` and `ccp_encoders.pkl` in the app folder.")
    else:
        st.success("Saved model loaded (ccp_rf_model.pkl).")

    # ---------- Build input form automatically from dataframe ----------
    st.subheader("Enter customer details for prediction")
    sample = df.copy()
    # If sample has non-numeric TotalCharges as string, try to convert
    for c in sample.columns:
        # ignore Churn
        if c == "Churn":
            continue
        # convert numeric-like strings
        if sample[c].dtype == object:
            # try to convert to numeric safely
            try:
                sample[c] = pd.to_numeric(sample[c])
            except:
                pass

    # Build form
    with st.form("predict_form"):
        left, right = st.columns(2)
        user_input = {}
        col_list = [c for c in sample.columns if c != "Churn"]
        # show a max of 12 columns to avoid giant forms; if many cols, user can pick subset
        if len(col_list) > 20:
            st.info("Dataset has many columns. We'll offer a subset for prediction. Use 'Select columns' to choose features.")
            selected_cols = st.multiselect("Select features to use for prediction", col_list, default=col_list[:12])
            col_list = selected_cols
        for i, col in enumerate(col_list):
            # choose numeric vs categorical
            if sample[col].dtype.kind in "biufc":  # numeric types
                min_val = float(sample[col].min())
                max_val = float(sample[col].max())
                mean_val = float(sample[col].mean())
                with (left if i%2==0 else right):
                    user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=mean_val, format="%f")
            else:
                options = list(sample[col].astype(str).unique())
                with (left if i%2==0 else right):
                    user_input[col] = st.selectbox(col, options)

        submit_pred = st.form_submit_button("🔮 Predict")

    # ---------- On Predict ----------
    if submit_pred:
        if model is None:
            st.error("No model available. Train a model above or provide saved model files.")
        else:
            # Need to construct a feature vector in same ordering used by model training
            # We'll attempt to use training X columns order (if available), otherwise use current col_list
            try:
                model_input_cols = list(model.feature_names_in_)
            except Exception:
                model_input_cols = col_list

            # Build input_df with model_input_cols order; fill missing with zeros or mean
            input_dict = {}
            for c in model_input_cols:
                if c in user_input:
                    input_dict[c] = user_input[c]
                else:
                    # fill default: if df has col, use mean / mode; else 0
                    if c in sample.columns:
                        if sample[c].dtype.kind in "biufc":
                            input_dict[c] = float(sample[c].mean())
                        else:
                            input_dict[c] = str(sample[c].mode().iloc[0])
                    else:
                        input_dict[c] = 0

            input_df = pd.DataFrame([input_dict])

            # Encode categoricals using saved encoders (if present)
            for col in input_df.columns:
                if col in encoders:
                    try:
                        # if input is numeric string but encoder expects categories, convert to str
                        input_df[col] = encoders[col].transform(input_df[col].astype(str))
                    except Exception as e:
                        # try to map using encoder classes; if unknown category, use -1
                        val = str(input_df[col].iloc[0])
                        classes = list(encoders[col].classes_)
                        if val in classes:
                            input_df[col] = encoders[col].transform([val])
                        else:
                            # add unknown handling
                            input_df[col] = -1

            # Ensure numeric dtypes for model
            for c in input_df.columns:
                try:
                    input_df[c] = pd.to_numeric(input_df[c])
                except:
                    pass

            # Reorder columns to model_input_cols
            input_df = input_df[model_input_cols]

            # Prediction
            pred = model.predict(input_df)[0]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1]
            else:
                prob = float(pred)

            # Result display
            st.subheader("Prediction Result")
            if pred == 1:
                st.error(f"❌ Customer is likely to **CHURN** (probability {prob:.2f})")
            else:
                st.success(f"✅ Customer is **NOT** likely to churn (probability {prob:.2f})")

            # ---------- Plotly Gauge for probability ----------
            try:
                import plotly.graph_objects as go
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if prob>0.5 else "green"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                # fallback to progress bar
                st.progress(min(max(float(prob),0.0),1.0))

            # ---------- Customer summary ----------
            st.subheader("Customer Input Summary")
            st.json(input_dict)


            st.subheader("📊 Feature Importance")
            st.image("feature_importance.png")
