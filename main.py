import os
import uuid
import gdown
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from keras.preprocessing.image import load_img, img_to_array  # type: ignore

app = Flask(__name__)
CORS(app)  # Enable CORS for React support

# Google Drive model file ID
MODEL_ID = "1aQ4-X0HCMkFtj7ZBfcAchOQoQe8rvGa7"

# Paths for models
H5_MODEL_PATH = 'cancer_model_catch.h5'
MODEL_PATH = os.getenv('MODEL_PATH', 'cancer_model_catch.tflite')

# Uploads folder
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.path.join(os.getcwd(), 'uploads'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download .h5 model from Google Drive
def download_model_from_drive(model_id, output_path):
    try:
        print(f"Downloading model with ID {model_id}...")
        gdown.download(f"https://drive.google.com/uc?export=download&id={model_id}", output_path, quiet=False)
        print(f"Model downloaded at {output_path}")
    except Exception as e:
        print(f"Download error: {e}")

# Convert .h5 to .tflite
def convert_h5_to_tflite(h5_path, tflite_path):
    try:
        model = tf.keras.models.load_model(h5_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Converted model to {tflite_path}")
    except Exception as e:
        print(f"Conversion error: {e}")

# Check and prepare TFLite model
if not os.path.exists(MODEL_PATH):
    print(f"{MODEL_PATH} not found. Downloading and converting model...")
    if not os.path.exists(H5_MODEL_PATH):
        download_model_from_drive(MODEL_ID, H5_MODEL_PATH)
    convert_h5_to_tflite(H5_MODEL_PATH, MODEL_PATH)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prediction labels
class_names = [
    'Carcinoma in Situ', 'Commissural Squamous cell carcinoma', 'Gum cancer',
    'Mucoepidermoid Carcinoma', 'Oral Cancer', 'Oral Lichen Planus',
    'Oral Tumors', 'This is not Oral related Oral Images'
]

# Predict function
def predict_cancer(image_path, confidence_threshold=0.80):
    try:
        img = load_img(image_path, target_size=(256, 256))
        img_arr = img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_arr)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_index = int(np.argmax(output_data))
        confidence = float(np.max(output_data))

        if confidence < confidence_threshold:
            return -1, confidence

        return class_names[predicted_index], confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

# API: Predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(file_path)
        result, confidence = predict_cancer(file_path)

        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        if result == -1:
            return jsonify({
                'class': "This photo may not be compatible. It seems non-cancerous, another object, or has scale/quality issues."
            })

        return jsonify({
            'class': result,
            'confidence': round(confidence * 100, 2),
            'image_url': f"/uploads/{unique_filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve uploaded image
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Root route
@app.route('/')
def home():
    return "Welcome to the Oral Cancer Prediction API!"

# Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
