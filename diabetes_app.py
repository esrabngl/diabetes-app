import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle (diabetes.csv dosyasını proje klasörüne koyun)
data = pd.read_csv("diabetes.csv")

# Streamlit başlığı
st.title("Diabetes Prediction App")

# Dataset'i göster
st.subheader("Dataset Preview")
st.write(data.head())

# Dataset istatistikleri
st.subheader("Data Exploration")
st.write("Basic Statistics of the Dataset")
st.write(data.describe())

# Eksik değer kontrolü
st.write("Missing Values:")
st.write(data.isnull().sum())

# Histogram
st.subheader("Feature Distribution")
column = st.selectbox("Select a column to visualize", data.columns)
fig, ax = plt.subplots()
sns.histplot(data[column], kde=True, ax=ax)
st.pyplot(fig)

# Veri işleme
st.subheader("Data Preprocessing")
data = data.replace(0, np.nan)  # Eksik değerleri NaN ile değiştir
data.fillna(data.median(), inplace=True)  # Eksik değerleri medyan ile doldur
st.write("Data after preprocessing:")
st.write(data.head())

# Eğitim ve test seti
st.subheader("Train a Model")
if "Outcome" in data.columns:
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model seçimi ve eğitimi
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model sonuçları
    st.write("Model Performance:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Tahmin arayüzü
    st.subheader("Make Predictions")
    inputs = {}
    for column in X.columns:
        inputs[column] = st.number_input(f"Enter value for {column}:", min_value=0.0, max_value=100.0, step=0.1)
    inputs_df = pd.DataFrame([inputs])
    if st.button("Predict"):
        prediction = model.predict(inputs_df)
        st.write("Prediction Result:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
else:
    st.write("Please ensure the dataset has an 'Outcome' column for predictions.")
