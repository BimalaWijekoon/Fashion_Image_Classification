import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

def load_dataset(data_dir, img_size=(28, 28), batch_size=32):
    """
    Load train, test, and validation datasets from the created directory structure
    """
    # Create data generators with proper augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    # Load datasets
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="grayscale"
    )

    val_generator = test_val_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),  # Changed from "validate" to match dataset script
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="grayscale"
    )

    test_generator = test_val_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="grayscale"
    )

    return train_generator, val_generator, test_generator

def build_model(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

def train_model(model, train_gen, val_gen, epochs=15):
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint]
    )
    return history

def main():
    # Path to dataset directory
    data_dir = "dataset"  # Update this path if needed
    
    # Load dataset
    train_gen, val_gen, test_gen = load_dataset(data_dir)
    
    # Model parameters
    input_shape = train_gen.image_shape
    num_classes = train_gen.num_classes
    
    # Build and compile model
    model = build_model(input_shape, num_classes)
    compile_model(model)
    
    # Train model
    history = train_model(model, train_gen, val_gen)
    
    # Evaluate on test set
    best_model = tf.keras.models.load_model('best_model.h5')
    test_loss, test_acc = best_model.evaluate(test_gen)
    print(f"\nTest accuracy: {test_acc:.2%}")
    
    # Save final model
    model.save("fashion_mnist_final_model.h5")

if __name__ == "__main__":
    main()