import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.title("🛒 Grocery Image Recognition Chatbot")

# FIX HERE
model = tf.keras.models.load_model("grocery_model.h5")

with open("classes.json") as f:
    class_names = json.load(f)

with open("grocery_database.json") as f:
    grocery_db = json.load(f)


def predict_image(image):

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    preds = model.predict(img)

    index = np.argmax(preds)

    label = class_names[index]
    confidence = preds[0][index]

    return label, confidence


uploaded = st.file_uploader("Upload Grocery Image")

if uploaded:

    image = Image.open(uploaded)

    st.image(image,width=300)

    product,confidence = predict_image(image)

    st.subheader("Detected Product")
    st.write(product)

    st.write("Confidence:",round(confidence,2))

    if product in grocery_db:

        st.subheader("Description")
        st.write(grocery_db[product]["description"])

        st.subheader("Nutrition")
        st.write(grocery_db[product]["nutrition"])