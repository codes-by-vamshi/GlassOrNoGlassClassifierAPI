import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Load the trained MobileNetV2 model
model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'mobilenet_glasses_classifier.h5'))

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # Preprocess the image for MobileNetV2
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    img = Image.open(request.files['file'])
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    print(predictions[0])
    class_idx = np.argmax(predictions[0])
    print(class_idx)
    class_labels = ['With glasses', 'Without glasses']
    return jsonify({'prediction': class_labels[class_idx]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

