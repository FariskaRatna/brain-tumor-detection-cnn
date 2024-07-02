import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import time
fig = plt.figure()
from cnn_model import cnn_model

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Bag Classifier')

st.markdown("Welcome to this simple web application that classifies bags. The bags are classified into six different classes namely: Backpack, Briefcase, Duffle, Handbag and Purse.")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)

def predict(image_path):
    classifier_model='https://github.com/FariskaRatna/brain-tumor-detection-cnn/releases/download/v1/best_23cnn.weights.h5'
    IMAGE_SHAPE=(512,512,3)
    model = cnn_model()
    model.load_weights(classifier_model)
    image = Image.open(image_path)
    test_image = image.resize((IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
        "glioma_tumor",
        "no_tumor", 
        "meningioma_tumor", 
        "pituitary_tumor"
    ]
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
#     results = {
#         "glioma_tumor": 0, 
#         "no_tumor": 0, 
#         "meningioma_tumor": 0, 
#         "pituitary_tumor": 0
#     }
    
    results = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence."
    return results

if __name__ == "__main__":
    main()