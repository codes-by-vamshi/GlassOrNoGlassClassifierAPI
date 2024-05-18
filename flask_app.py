from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('./resnet_glasses_classifier.h5')

def preprocess_image(img):
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    img = Image.open(request.files['file'])
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_labels = ['With out glasses', 'With glasses']
    return jsonify({'prediction': class_labels[class_idx]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
