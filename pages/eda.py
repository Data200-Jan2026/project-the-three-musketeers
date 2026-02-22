import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA Dashboard")

st.title("Exploratory Data Analysis")
st.write("Visualizing the relationships in the Diabetes Dataset.")

# Load Data
@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/diabetes_prediction_dataset.csv')
    except:
        return None

df = load_data()

if df is not None:
    # Show Raw Data
    if st.checkbox("Show Raw Data"):
        st.write(df.head())

    # --- Visualizations ---

    # 1. Target Distribution
    st.subheader("1. How many people have diabetes?")
    st.write("Distribution of Target Variable (0 = No, 1 = Yes)")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='diabetes', data=df, palette='viridis', ax=ax1)
    st.pyplot(fig1)

    # 2. Correlation Matrix
    st.subheader("2. Which factors matter most?")
    st.write("Correlation Matrix showing relationships between features.")

    # Encode for correlation just for this view
    df_encoded = df.copy()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_encoded['gender'] = le.fit_transform(df_encoded['gender'])
    df_encoded['smoking_history'] = le.fit_transform(df_encoded['smoking_history'])

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    # 3. Glucose vs Diabetes
    st.subheader("3. Blood Glucose Levels by Diagnosis")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='diabetes', y='blood_glucose_level', data=df, palette='Set2', ax=ax3)
    st.pyplot(fig3)

else:
    st.error("File 'diabetes_prediction_dataset.csv' not found.")
