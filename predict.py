import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from cnn_model import cnn_model

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

image = "dataset//Testing//pituitary_tumor//image(10).jpg"
predictions = predict(image)
print(predictions)