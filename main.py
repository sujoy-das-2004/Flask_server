# import os
# import uuid
# import gdown
# import numpy as np
# import tensorflow as tf  # Use TensorFlow for conversion
# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# from keras.preprocessing.image import load_img, img_to_array  # type: ignore

# app = Flask(__name__)
# CORS(app)  # Enable CORS for React support

# # Google Drive model file ID (replace with your model's ID)
# MODEL_ID = "1Yo5HXLOEytYpqDVavWfbAtHRnLha5Jel"  # Your Google Drive file ID
# MODEL_PATH = os.getenv('MODEL_PATH', 'cancer_model.tflite')  # Model file path for TFLite
# H5_MODEL_PATH = 'cancer_model.h5'  # Path to the .h5 model (if applicable)

# # Use absolute path for UPLOAD_FOLDER
# UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.path.join(os.getcwd(), 'uploads'))  # Absolute path for uploads
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Ensure upload folder exists and is writable
# if not os.path.exists(UPLOAD_FOLDER):
#     print(f"Creating upload folder at {UPLOAD_FOLDER}")
#     os.makedirs(UPLOAD_FOLDER)
# else:
#     print(f"Upload folder already exists at {UPLOAD_FOLDER}")

# # Function to download .h5 model from Google Drive
# def download_model_from_drive(model_id, output_path):
#     try:
#         print(f"Downloading model from Google Drive with ID {model_id}...")
#         gdown.download(f"https://drive.google.com/uc?export=download&id={model_id}", output_path, quiet=False)
#         print(f"Model downloaded successfully at {output_path}")
#     except Exception as e:
#         print(f"Error downloading model: {e}")

# # Function to convert .h5 model to .tflite
# def convert_h5_to_tflite(h5_model_path, tflite_model_path):
#     try:
#         # Load the .h5 model
#         model = tf.keras.models.load_model(h5_model_path)
#         print("Model loaded successfully!")

#         # Convert the model to TensorFlow Lite format
#         converter = tf.lite.TFLiteConverter.from_keras_model(model)
#         tflite_model = converter.convert()

#         # Save the converted model
#         with open(tflite_model_path, 'wb') as f:
#             f.write(tflite_model)

#         print(f"Model successfully converted to {tflite_model_path}!")
#     except Exception as e:
#         print(f"Error during conversion: {e}")

# # Check if .tflite model exists, otherwise convert .h5 model to .tflite
# if not os.path.exists(MODEL_PATH):
#     print(f"{MODEL_PATH} not found, attempting to download and convert model...")
    
#     if not os.path.exists(H5_MODEL_PATH):  # If the .h5 model doesn't exist locally
#         download_model_from_drive(MODEL_ID, H5_MODEL_PATH)
        
#     # Convert the downloaded .h5 model to .tflite
#     convert_h5_to_tflite(H5_MODEL_PATH, MODEL_PATH)

# # Load the TFLite model using TensorFlow Lite Interpreter
# interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
# interpreter.allocate_tensors()

# # Get input and output tensor details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Class names for prediction
# class_names = [
#     'Carcinoma in Situ', 'Commissural Squamous cell carcinoma', 'Gum cancer',
#     'Mucoepidermoid Carcinoma', 'Oral Cancer', 'Oral Lichen Planus', 'Oral Tumors'
# ]

# # Helper function to predict cancer using TFLite
# def predict_cancer(image_path):
#     try:
#         IMAGE_SIZE = 256
#         img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))  # Resize image to match model input
#         img_arr = img_to_array(img) / 255.0  # Normalize the image
#         img_arr = np.expand_dims(img_arr, axis=0)

#         # Set the tensor for input
#         interpreter.set_tensor(input_details[0]['index'], img_arr)

#         # Run inference
#         interpreter.invoke()

#         # Get the output tensor
#         output_data = interpreter.get_tensor(output_details[0]['index'])

#         # Get the predicted class and confidence
#         predicted_class_index = int(np.argmax(output_data, axis=1)[0])
#         confidence_score = float(np.max(output_data))

#         predicted_class = class_names[predicted_class_index]
#         return predicted_class, confidence_score
#     except Exception as e:
#         print("Prediction error:", e)
#         return None, None

# # API Route for Prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     print(f"Received file: {file.filename}")  # Log the filename

#     # Generate a unique filename for the uploaded file
#     unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
#     file_location = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

#     try:
#         # Save the file temporarily
#         file.save(file_location)
#         print(f"File saved at: {file_location}")  # Log after saving the file

#         # Predict
#         result, confidence = predict_cancer(file_location)

#         if result is None:
#             return jsonify({'error': 'Prediction failed'}), 500

#         # Optionally delete file after prediction
#         # os.remove(file_location)

#         # Return the image URL so that the frontend can access it
#         image_url = f"/uploads/{unique_filename}"

#         print(f"Prediction result: {result}, Confidence: {confidence}")
#         print(f"Image URL: {image_url}")

#         return jsonify({
#             'class': result,
#             'confidence': round(confidence * 100, 2),  # Convert confidence to percentage
#             'image_url': image_url  # Send the URL of the uploaded image
#         })
#     except Exception as e:
#         print("Error:", e)
#         return jsonify({'error': str(e)}), 500

# # Serve uploaded images (if needed)
# @app.route('/uploads/<filename>')
# def get_uploaded_file(filename):
#     print(f"Serving file: {filename}")
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# # Root Route for Home Page
# @app.route('/')
# def home():
#     return "Welcome to the Oral Cancer Prediction API!"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)



import os
import uuid
import gdown
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
CORS(app)  # Enable CORS for React support

# Google Drive model file ID
MODEL_ID = "1__7RCc4JBIGz5W8zujUVtakIvHkvW5GE"  # Replace with your actual Google Drive file ID
MODEL_PATH = os.getenv('MODEL_PATH', 'cancer_model_catch.tflite')  # Path for TFLite model
H5_MODEL_PATH = 'cancer_model_catch.h5'  # Path for .h5 model
#https://drive.google.com/file/d/1__7RCc4JBIGz5W8zujUVtakIvHkvW5GE/view?usp=sharing
# Use absolute path for uploads
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.path.join(os.getcwd(), 'uploads'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to download .h5 model from Google Drive
def download_model_from_drive(model_id, output_path):
    try:
        print(f"Downloading model from Google Drive with ID {model_id}...")
        gdown.download(f"https://drive.google.com/uc?export=download&id={model_id}", output_path, quiet=False)
        print(f"Model downloaded successfully at {output_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")

# Function to convert .h5 model to .tflite
def convert_h5_to_tflite(h5_model_path, tflite_model_path):
    try:
        model = tf.keras.models.load_model(h5_model_path)
        print("Model loaded successfully!")

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Model successfully converted to {tflite_model_path}!")
    except Exception as e:
        print(f"Error during conversion: {e}")

# Check if TFLite model exists, otherwise download and convert .h5 model
if not os.path.exists(MODEL_PATH):
    print(f"{MODEL_PATH} not found, attempting to download and convert model...")
    if not os.path.exists(H5_MODEL_PATH):
        download_model_from_drive(MODEL_ID, H5_MODEL_PATH)
    convert_h5_to_tflite(H5_MODEL_PATH, MODEL_PATH)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names for prediction
class_names = [
    'Carcinoma in Situ', 'Commissural Squamous cell carcinoma', 'Gum cancer',
    'Mucoepidermoid Carcinoma', 'Oral Cancer', 'Oral Lichen Planus', 'Oral Tumors'
]

# Helper function to predict cancer using TFLite
def predict_cancer(image_path, confidence_threshold=0.958):
    try:
        IMAGE_SIZE = 256
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_arr = img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        # Set the tensor for input
        interpreter.set_tensor(input_details[0]['index'], img_arr)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get the predicted class and confidence
        predicted_class_index = int(np.argmax(output_data, axis=1)[0])
        confidence_score = float(np.max(output_data))

        # If confidence is below threshold, return -1
        if confidence_score < confidence_threshold:
            return -1, confidence_score  # Unrelated image
        
        predicted_class = class_names[predicted_class_index]
        return predicted_class, confidence_score
    except Exception as e:
        print("Prediction error:", e)
        return None, None

# API Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    print(f"Received file: {file.filename}")

    # Generate a unique filename
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_location = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        # Save the file temporarily
        file.save(file_location)
        print(f"File saved at: {file_location}")

        # Predict
        result, confidence = predict_cancer(file_location)

        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # If model returns -1, classify as "Not an oral cancer image"
        if result == -1:
            return jsonify({
                'class': "This photo may not be compatible. It seems non-cancerous, another object, or has scale/quality issues."
            })

        # Return prediction result
        return jsonify({
            'class': result,
            'confidence': round(confidence * 100, 2),
            'image_url': f"/uploads/{unique_filename}"
        })
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

# Serve uploaded images
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Root Route for Home Page
@app.route('/')
def home():
    return "Welcome to the Oral Cancer Prediction API!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

