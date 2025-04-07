import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"  # Set PlaidML as the Keras backend

import plaidml.keras
plaidml.keras.install_backend()  # Explicitly set PlaidML as the Keras backend

# Use a non-interactive backend for matplotlib to avoid Tcl/Tk dependency issues
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import numpy as np
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import pywt
from keras.preprocessing.image import ImageDataGenerator
import math
import cv2  # Import OpenCV for image resizing

# Helper function to preprocess images with DWT (used for comparison)
def preprocess_images_with_dwt(image_batch, level=1):
    batch_size, height, width, channels = image_batch.shape
    current_height, current_width = height, width

    # Adjust the shape based on the level of decomposition
    for _ in range(level):
        current_height //= 2
        current_width //= 2

    dwt_batch = np.zeros((batch_size, current_height, current_width, channels))
    for i in range(batch_size):
        for c in range(channels):
            coeff = image_batch[i, :, :, c]
            # Apply DWT for the given level
            for _ in range(level):
                LL, (LH, HL, HH) = pywt.dwt2(coeff, 'haar')
                coeff = LL
            dwt_batch[i, :, :, c] = coeff

    return dwt_batch

# Function to load and preprocess images
def load_images(data_directory, target_size=(256, 256), batch_size=10):
    data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    generator = data_gen.flow_from_directory(
        data_directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None
    )
    images = next(generator)  # Load a batch of images for testing
    return images

# Function to compute PSNR between two images
def compute_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0  # Since images are normalized to [0, 1]
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Main function to load model and perform evaluation
def main():
    model_path = 'autoencoder_model_rgb_with_cayley_dwt.h5'
    if not os.path.exists(model_path):
        print("Trained model not found. Please train the model first.")
        return

    # Load the trained autoencoder model
    autoencoder = load_model(model_path)
    print("Loaded trained model.")

    # Load and preprocess images
    data_directory = "./Medical-imaging-dataset"
    original_images = load_images(data_directory, target_size=(256, 256), batch_size=5)

    # Apply DWT preprocessing to match the input expected by the autoencoder
    dwt_images = preprocess_images_with_dwt(original_images, level=1)

    # Use the model to predict (reconstruct) the images
    reconstructed_images = autoencoder.predict(dwt_images)

    # Resize reconstructed images to match original image dimensions
    resized_reconstructed_images = np.zeros_like(original_images)
    for i in range(len(reconstructed_images)):
        for c in range(reconstructed_images.shape[3]):
            resized_reconstructed_images[i, :, :, c] = cv2.resize(
                reconstructed_images[i, :, :, c],
                (original_images.shape[2], original_images.shape[1]),
                interpolation=cv2.INTER_LINEAR
            )

    # Compute and display PSNR for each image
    for i in range(len(original_images)):
        original = original_images[i]
        reconstructed = resized_reconstructed_images[i]
        psnr_value = compute_psnr(original, reconstructed)
        print(f"Image {i+1}: PSNR = {psnr_value:.2f} dB")

if __name__ == "__main__":
    main()
