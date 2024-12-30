import os
import base64
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import img_to_array
import cv2

app = Flask(__name__)

# Load pre-trained model
model = tf.keras.models.load_model("model/digit_model.h5")

# Path for uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route for camera-captured image
@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'})

    # Decode the base64 image
    image_data = data['image'].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1) / 255.0

    # Make prediction
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    return jsonify({
        'prediction': int(predicted_label),
        'confidence': f"{np.max(prediction) * 100:.2f}%"
    })

# Predict route for uploaded image
@app.route('/predict_file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Preprocess the image
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        image = image.reshape(1, 28, 28, 1) / 255.0

        # Make prediction
        prediction = model.predict(image)
        predicted_label = np.argmax(prediction)

        return jsonify({
            'prediction': int(predicted_label),
            'confidence': f"{np.max(prediction) * 100:.2f}%"
        })

if __name__ == '__main__':
    app.run(debug=True)
