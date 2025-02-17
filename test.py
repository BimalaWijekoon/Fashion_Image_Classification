import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
    predictions = model.predict(img_array)[0]
    return predictions

# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if file_path:
        # Clear previous results
        for widget in results_frame.winfo_children():
            widget.destroy()
        
        # Display loading state
        loading_label.config(text="Analyzing image...")
        root.update()
        
        try:
            # Display the uploaded image
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img

            # Make prediction
            predictions = predict_image(file_path)
            
            # Hide loading text
            loading_label.config(text="")
            
            # Display results with progress bars
            for i, (class_name, confidence) in enumerate(zip(class_labels, predictions)):
                create_result_row(i, class_name, confidence)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            loading_label.config(text="")

def create_result_row(index, class_name, confidence):
    row_frame = tk.Frame(results_frame)
    row_frame.pack(fill='x', pady=2)
    
    # Class label
    label = tk.Label(row_frame, text=class_name, width=15, anchor='w')
    label.pack(side='left', padx=5)
    
    # Percentage label
    perc_label = tk.Label(row_frame, text=f"{confidence:.1%}", width=6)
    perc_label.pack(side='left')
    
    # Progress bar
    progress = ttk.Progressbar(row_frame, orient='horizontal', 
                             length=200, mode='determinate')
    progress['value'] = confidence * 100
    progress.pack(side='left', expand=True, fill='x')
    
    # Highlight top prediction
    if confidence == max(predictions):
        label.config(fg='#2ecc71', font=('Helvetica', 9, 'bold'))
        perc_label.config(fg='#2ecc71', font=('Helvetica', 9, 'bold'))
        progress.config(style='success.Horizontal.TProgressbar')
    else:
        label.config(fg='#34495e')
        progress.config(style='default.Horizontal.TProgressbar')

# Create the main GUI window
root = tk.Tk()
root.title("Fashion Classifier Pro")
root.geometry("600x700")
root.configure(bg='#ecf0f1')

# Custom styles
style = ttk.Style()
style.theme_use('clam')
style.configure('default.Horizontal.TProgressbar', 
               background='#bdc3c7', troughcolor='#ecf0f1')
style.configure('success.Horizontal.TProgressbar', 
               background='#2ecc71', troughcolor='#ecf0f1')

# Header
header_frame = tk.Frame(root, bg='#2ecc71')
header_frame.pack(fill='x', pady=(0, 20))
title_label = tk.Label(header_frame, text="Fashion Classifier Pro", 
                      font=('Helvetica', 16, 'bold'), bg='#2ecc71', fg='white')
title_label.pack(pady=10)

# Upload Button
upload_button = ttk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

# Image display area
image_label = tk.Label(root, bg='#ecf0f1')
image_label.pack(pady=10)

# Loading label
loading_label = tk.Label(root, text="", font=('Helvetica', 12), bg='#ecf0f1', fg='#e74c3c')
loading_label.pack()

# Results frame
results_frame = tk.Frame(root, bg='#ecf0f1')
results_frame.pack(fill='both', expand=True, padx=20, pady=10)

# Start the GUI event loop
root.mainloop()
