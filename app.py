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

# Treatment suggestions
suggestions = {
    'Tomato_Bacterial_spot': 'A plant with bacterial spot cannot be cured. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit.',
    'Tomato_Early_blight': 'Cure the plant quickly otherwise the disease can spread. Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable.',
    'Tomato_Late_blight': 'Spraying fungicides is the most effective way to prevent late blight. For conventional gardeners and commercial producers, protectant fungicides such as chlorothalonil (e.g., Bravo, Echo, Equus, or Daconil) and Mancozeb (Manzate) can be used.',
    'Tomato_Leaf_Mold': 'Baking soda solution: Mix 1 tablespoon baking soda and ½ teaspoon liquid soap such as Castile soap (not detergent) in 1 gallon of water. Spray liberally, getting top and bottom leaf surfaces and any affected areas.',
    'Tomato_Septoria_leaf_spot': 'Fungicides with active ingredients such as chlorothalonil, copper, or mancozeb will help reduce disease, but they must be applied before disease occurs as they can only provide preventative protection. They will not cure the plant. If the disease has spread, then remove the plants.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Aim a hard stream of water at infested plants to knock spider mites off the plants. Other options include insecticidal soaps, horticultural oils, or neem oil.',
    'Tomato__Target_Spot': 'Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spot in research trials.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Use a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum), as a soil application or through the drip irrigation system at transplanting of tomatoes or peppers.',
    'Tomato__Tomato_mosaic_virus': 'Remove all infected plants and destroy them. Do NOT put them in the compost pile, as the virus may persist in infected plant matter. Monitor the rest of your plants closely, especially those that were located near infected plants. Disinfect gardening tools after every use.',
    'Tomato_healthy': 'Your plant is healthy, there is no need to apply medicines. Please take care of your plants. If any disease occurs, then cure it fast and remove the infected leaves.'
}

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

# Read file content
def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

# Streamlit App
def main():
    st.title('Tomato Disease Classifier')

    # Embedding custom HTML
    st.markdown(read_file('index.html'), unsafe_allow_html=True)

    # Embedding custom CSS
    st.markdown(f'<style>{read_file("styles.css")}</style>', unsafe_allow_html=True)

    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file is not None:
        image = file.read()  # Read the uploaded file as bytes
        st.image(image, caption='Uploaded Image.', use_column_width=True)  # Display the uploaded image

        if st.button('Predict'):
            predicted_class, confidence = predict(image)
            st.write(f"Predicted Class: {classes[predicted_class]}")
            st.write(f"Confidence: {confidence:.2f}%")

            # Display treatment suggestion
            disease = classes[predicted_class]
            st.write("Treatment Suggestion:")
            st.write(suggestions[disease])

if __name__ == '__main__':
    main()
