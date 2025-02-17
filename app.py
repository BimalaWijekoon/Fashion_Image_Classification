# app.py
from flask import Flask, request, jsonify
from tensorflow import keras
from PIL import Image
import numpy as np
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes 


# Load the trained model
MODEL_PATH = "fashion_mnist_final_model.h5"
model = keras.models.load_model(MODEL_PATH)

# Class labels for Fashion MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Function to preprocess the image for the model
def preprocess_image(image):
    img = Image.open(image).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)  # Convert to numpy array
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
    img_array = img_array.astype("float32") / 255.0  # Normalize
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            img_array = preprocess_image(file)
            predictions = model.predict(img_array)[0]
            top_prediction_idx = np.argmax(predictions)
            confidence = predictions[top_prediction_idx]
            predicted_class = class_labels[top_prediction_idx]
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': float(confidence)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

