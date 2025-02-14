import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
MODEL_PATH = "fashion_mnist_final_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels for Fashion MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Function to preprocess the image for the model
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)  # Convert to numpy array
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
    img_array = img_array.astype("float32") / 255.0  # Normalize
    return img_array

# Function to make predictions
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_labels[predicted_class], confidence

# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if file_path:
        # Display the uploaded image
        img = Image.open(file_path)
        # Line 43 in upload_image function
        img = img.resize((200, 200), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img  # Keep a reference to avoid garbage collection

        # Make prediction
        predicted_class, confidence = predict_image(file_path)
        result_label.config(text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}")

# Create the main GUI window
root = tk.Tk()
root.title("Fashion MNIST Classifier")
root.geometry("400x400")
root.configure(bg="#f0f0f0")

# Title label
title_label = tk.Label(
    root,
    text="Fashion MNIST Classifier",
    font=("Helvetica", 16, "bold"),
    bg="#f0f0f0"
)
title_label.pack(pady=10)

# Upload button
upload_button = tk.Button(
    root,
    text="Upload Image",
    command=upload_image,
    font=("Helvetica", 12),
    bg="#4CAF50",
    fg="white"
)
upload_button.pack(pady=10)

# Image display label
image_label = tk.Label(root, bg="#f0f0f0")
image_label.pack(pady=10)

# Result label
result_label = tk.Label(
    root,
    text="Prediction: None\nConfidence: 0%",
    font=("Helvetica", 12),
    bg="#f0f0f0"
)
result_label.pack(pady=10)

# Run the GUI
root.mainloop()