import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("plant_disease_model.h5")

class_labels = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

remedies = {
    "Pepper__bell___Bacterial_spot": "Use copper-based bactericides and remove infected leaves to prevent spread. Ensure good air circulation around plants.",
    "Pepper__bell___healthy": "Plants look healthy. Maintain regular watering and monitor for pests or diseases.",
    "Potato___Early_blight": "Apply fungicides containing chlorothalonil or copper regularly. Remove and destroy infected foliage.",
    "Potato___Late_blight": "Use fungicides like mancozeb early and often. Avoid overhead watering and remove infected plants immediately.",
    "Potato___healthy": "Healthy plants need regular care; water consistently and check for pests frequently.",
    "Tomato_Bacterial_spot": "Spray copper-based sprays and avoid working with wet plants to reduce spread. Remove infected plant parts.",
    "Tomato_Early_blight": "Use fungicides like chlorothalonil and remove debris from around plants. Rotate crops annually.",
    "Tomato_Late_blight": "Apply fungicides such as mancozeb early in the season and remove affected plants promptly.",
    "Tomato_Leaf_Mold": "Improve airflow by pruning and apply fungicides containing copper or chlorothalonil.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves and apply fungicides regularly. Use drip irrigation to keep foliage dry.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray insecticidal soap or neem oil regularly. Introduce natural predators like ladybugs.",
    "Tomato__Target_Spot": "Remove infected leaves and apply appropriate fungicides. Avoid overhead irrigation to reduce spread.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whitefly populations with insecticides and remove infected plants to prevent spread.",
    "Tomato__Tomato_mosaic_virus": "Use resistant varieties and disinfect tools regularly. Remove infected plants promptly.",
    "Tomato_healthy": "Healthy plants; continue good care practices and monitor regularly for early signs of disease."
}


st.title("Plant Disease Detector")
st.write("Upload a leaf image to detect disease and get remedies.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)*100

    
    
   
    if confidence < 70:
        st.warning('The image you uploaded was not able to be reconginzed my the algorithm')
    
    else:
        st.write(f"Confidence: {confidence:.2f}%")
        st.subheader(f"Prediction: **{predicted_class}**")
        st.write(" Remedy:", remedies[predicted_class])

        if confidence < 90:
            st.warning('There can be a chance that what the algoritm has predicted is wrong. \n Please upload a better picture or double check')