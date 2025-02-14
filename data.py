import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist

def create_directory_structure():
    """Create directory structure for the dataset"""
    base_dir = "dataset"
    splits = ["train", "test", "val"]
    classes = list(range(10))  # 0-9 for Fashion MNIST classes
    
    for split in splits:
        for cls in classes:
            path = os.path.join(base_dir, split, str(cls))
            os.makedirs(path, exist_ok=True)
    return base_dir

def save_images(data, labels, split_dir, class_names):
    """Save images to appropriate directories"""
    for i, (image, label) in enumerate(zip(data, labels)):
        class_dir = os.path.join(split_dir, str(label))
        filename = os.path.join(class_dir, f"image_{i:05d}.png")
        plt.imsave(filename, image, cmap='gray')

def download_and_split_dataset():
    # Load Fashion MNIST dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Split train into train and validation (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create directory structure
    base_dir = create_directory_structure()
    
    # Define class names for Fashion MNIST
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    # Save datasets
    print("Saving training set...")
    save_images(X_train, y_train, os.path.join(base_dir, "train"), class_names)
    
    print("Saving validation set...")
    save_images(X_val, y_val, os.path.join(base_dir, "val"), class_names)
    
    print("Saving test set...")
    save_images(X_test, y_test, os.path.join(base_dir, "test"), class_names)
    
    print("\nDataset structure created:")
    print(f"Train: {len(X_train)} images")
    print(f"Validation: {len(X_val)} images")
    print(f"Test: {len(X_test)} images")
    print(f"Total: {len(X_train)+len(X_val)+len(X_test)} images")

if __name__ == "__main__":
    download_and_split_dataset()