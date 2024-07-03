import streamlit as st
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('Tomato_60.h5')

# Define your class names
classes = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Function to preprocess the image
def preprocess_image(image):
    image = tf.image.decode_image(image, channels=3)  # Decode the image
    image = tf.image.resize(image, (256, 256))        # Resize the image
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32
    image = tf.expand_dims(image, axis=0)            # Add batch dimension
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
    image = file.read()  # Read the uploaded file as bytes
    st.image(image, caption='Uploaded Image.', use_column_width=True)  # Display the uploaded image

    if st.button('Predict'):
        predicted_class, confidence = predict(image)
        st.write(f"Predicted Class: {classes[predicted_class]}")
        st.write(f"Confidence: {confidence:.2f}%")
