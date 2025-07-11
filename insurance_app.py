import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("insurance.csv")
df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = df.drop(columns='charges')
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)

st.title("💸 Insurance Cost Prediction App")

st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.radio("Sex", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.7)
children = st.sidebar.slider("Children", 0, 5, 0)
smoker = st.sidebar.radio("Smoker", ["Yes", "No"])
region = st.sidebar.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])


sex = 1 if sex == "Female" else 0
smoker = 1 if smoker == "No" else 0


region_dict = {
    "Southeast": 0,
    "Southwest": 1,
    "Northeast": 2,
    "Northwest": 3
}
region = region_dict[region]


input_data = np.asarray([[age, sex, bmi, children, smoker, region]])

if st.button("Predict Insurance Cost"):
    prediction = reg.predict(input_data)
    st.success(f"The estimated insurance cost is 💲{prediction[0]:,.2f}")
