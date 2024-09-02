import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# Load the model
model = tf.keras.models.load_model('C:/Kritika/PotatoDiseaseFinal/models/model1.keras')

# Define the class names
class_names = ['early blight', 'late blight', 'healthy']

# Define the prediction function
def predict(img_array):
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Streamlit app
st.title('Potato Disease Classification')

st.write("Upload a potato leaf image to classify its disease.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file).resize((256, 256))
    img_array = np.array(img) / 255.0  # Normalize image array

    # Make a prediction
    predicted_class, confidence = predict(img_array)

    # Display the image and prediction
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {predicted_class}')
    st.write(f'Confidence: {confidence}%')
