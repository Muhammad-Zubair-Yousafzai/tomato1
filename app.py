import streamlit as st
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('Tomato_60.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = tf.image.resize(image, (256, 256))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    return predicted_class, confidence

# Streamlit App
st.title('Tomato Disease Classifier')
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = tf.image.decode_image(file.read(), channels=3)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        predicted_class, confidence = predict(image)
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence:.2f}%")
