#trying to deploy to the cloud
#import libraries
import streamlit as st
import joblib
import pandas as pd

#Load trained model and scaler
model = joblib.load("iris_logistic_regression_model.joblib")
scaler = joblib.load("iris_scaler.joblib")

#App title
st.title("Iris Species Predictor")
st.write("Adjust the sliders to predict the iris species.")

#User input sliders
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

#Create input DataFrame
input_df = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=["SepalLengthCm",
             "SepalWidthCm",
             "PetalLengthCm",
             "PetalWidthCm"
    ]
)

#Scale input
scaled_input = scaler.transform(input_df)

#Make prediction
prediction = model.predict(scaled_input)[0]

#Prediction probabilities
probability = model.predict_proba(scaled_input)[0]
confidence = max(probability)

#Display result
st.subheader("Prediction")
st.write(f"Predicted iris species: **{prediction}**")

st.subheader("Prediction Probabilities")
for species, p in zip(model.classes_, probability):
    st.write(f"{species}: {p:.2%}")

st.subheader("Model confidence")
st.write("Confidence in the predicted class:")
st.progress(confidence)
st.write(f"{confidence:.2%}")

#Interpretation
st.caption ("Model confidence represents the highest predicted class probability and reflects the model's certainty for the given input.")