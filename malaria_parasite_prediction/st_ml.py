import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras

# Load the pre-trained model
model_file = "c:/Users/danus/Downloads/malaria/malaria_model.h5"
model = keras.models.load_model(model_file)
input_shape = (124, 124)

categories = ["Parasitized", "Uninfected"]

def prepare_img(img):
    resized = cv2.resize(img, input_shape, interpolation=cv2.INTER_AREA)
    imgresult = np.expand_dims(resized, axis=0)
    imgresult = imgresult / 255
    return imgresult

def main():
    st.title("Malaria Parasite Detection App")
    st.sidebar.header("Choose an Image")

    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imread(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image for model prediction
        img_for_model = prepare_img(image)

        # Make prediction
        result = model.predict(img_for_model)

        # Get the predicted category
        predicted_category = categories[np.argmax(result)]

        st.sidebar.subheader("Prediction Result:")
        st.sidebar.text(f"The image is predicted to be: {predicted_category}")

if __name__ == "__main__":
    main()
