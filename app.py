import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('Tomato_60.h5')
    return model

model = load_model()

# Define classes for classification
classes = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
           'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
           'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
           'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
           'Tomato_healthy']

# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict the class and get treatment suggestions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    return predicted_class, prediction

def get_treatment_suggestions(predicted_class):
    suggestions = {
        'Tomato_Bacterial_spot': 'Treatment suggestions for Tomato Bacterial Spot...',
        'Tomato_Early_blight': 'Treatment suggestions for Tomato Early Blight...',
        'Tomato_Late_blight': 'Treatment suggestions for Tomato Late Blight...',
        'Tomato_Leaf_Mold': 'Treatment suggestions for Tomato Leaf Mold...',
        'Tomato_Septoria_leaf_spot': 'Treatment suggestions for Tomato Septoria Leaf Spot...',
        'Tomato_Spider_mites_Two_spotted_spider_mite': 'Treatment suggestions for Two-Spotted Spider Mite...',
        'Tomato__Target_Spot': 'Treatment suggestions for Tomato Target Spot...',
        'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Treatment suggestions for Tomato Yellow Leaf Curl Virus...',
        'Tomato__Tomato_mosaic_virus': 'Treatment suggestions for Tomato Mosaic Virus...',
        'Tomato_healthy': 'No treatment suggestions needed for healthy tomatoes.'
    }
    return suggestions.get(predicted_class, 'No treatment suggestions available.')

# Streamlit app
def main():
    st.title('Tomato Disease Classification and Treatment Suggestions')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('Predict'):
            predicted_class, prediction = predict(uploaded_file)
            treatment_suggestions = get_treatment_suggestions(predicted_class)

            st.subheader('Prediction')
            st.write(f"Predicted class: {predicted_class}")
            st.write(f"Prediction probabilities: {prediction}")

            st.subheader('Treatment Suggestions')
            st.write(treatment_suggestions)

if __name__ == '__main__':
    main()
