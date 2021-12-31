import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

model_age = tf.keras.models.load_model("best_model_age_tunning_2.h5")
model_gender = tf.keras.models.load_model("best_model_gender_tunning_2.h5") 

map_dict_gender = {
    0 : 'female',
    1 : 'male'
}

map_dict_age = {
    0 : '0-2',
    1 : '4-6',
    2 : '8-13',
    3 : '15-20',
    4 : '25-32',
    5 : '38-43',
    6 : '48-53',
    7 : '60+'
}

def processing(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    opencv_image = cv2.resize(opencv_image,(227,227))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")
    # data = np.asarray(opencv_image)
    test = np.asarray(opencv_image)
    img_reshape = test[np.newaxis,...]
    return img_reshape

def predict_age(model_age, img):
    prediction_age = model_age.predict(img).argmax()
    return prediction_age


def predict_gender(model_gender, img):
    # resized = mobilenet_v2_preprocess_input(resized)
    prediction_gender = model_gender.predict(img).argmax()
    return prediction_gender

### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")
col1, col2 = st.columns(2)
if uploaded_file is not None:
    with col1:img = processing(uploaded_file)
    age = predict_age(model_age, img)
    gender = predict_gender(model_gender,img)

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred: 
        with col2:st.header("Predicted Label for the image is {}".format(map_dict_gender[gender]))
        with col2:st.header("Predicted Label for the image is {}".format(map_dict_age[age]))
