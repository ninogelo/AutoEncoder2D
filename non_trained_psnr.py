import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"  # Set PlaidML as the Keras backend

import plaidml.keras
plaidml.keras.install_backend()  # Explicitly set PlaidML as the Keras backend

# Use a non-interactive backend for matplotlib to avoid Tcl/Tk dependency issues
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import numpy as np
import matplotlib.pyplot as plt
import pywt
from keras.preprocessing.image import ImageDataGenerator
import math

# Helper function to preprocess images with DWT
def preprocess_images_with_dwt_only(image_batch):
    batch_size, height, width, channels = image_batch.shape
    dwt_batch = np.zeros((batch_size, height // 2, width // 2, channels))
    for i in range(batch_size):
        for c in range(channels):
            channel = image_batch[i, :, :, c]
            LL, (LH, HL, HH) = pywt.dwt2(channel, 'haar')
            dwt_batch[i, :, :, c] = LL
    return dwt_batch

# Function to load images from directory
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
def compute_psnr(original, transformed):
    mse = np.mean((original - transformed) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0  # Since images are normalized to [0, 1]
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Function to calculate and plot coefficient differences and PSNR for "DWT Only"
def calculate_dwt_only_coefficient_differences_and_psnr(original_images):
    dwt_only_images = preprocess_images_with_dwt_only(original_images)

    differences = []
    psnr_values = []

    for i in range(len(dwt_only_images)):
        original_image = original_images[i, :, :, :]
        LL = dwt_only_images[i, :, :, :]

        # Resize original to match LL size
        original_resized = original_image[::2, ::2, :]

        # Calculate coefficient differences
        diff = np.abs(original_resized - LL)
        differences.append(np.mean(diff))

        # Calculate PSNR
        psnr_value = compute_psnr(original_resized, LL)
        psnr_values.append(psnr_value)

    # Print summary of coefficient differences and PSNR for each image
    for i, (diff, psnr_value) in enumerate(zip(differences, psnr_values)):
        print(f"DWT Only - Image {i+1}: Mean coefficient difference = {diff}, PSNR = {psnr_value:.2f} dB")

    # Plot the mean coefficient differences
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(differences) + 1), differences, color='lightblue')
    plt.xlabel('Image Index')
    plt.ylabel('Mean Coefficient Difference')
    plt.title('Mean Coefficient Differences for DWT Only Transformation')
    plt.savefig("dwt_only_coefficient_differences.png")  # Save the plot as an image file
    print("DWT Only coefficient differences plot saved as 'dwt_only_coefficient_differences.png'.")

    # Plot the PSNR values
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(psnr_values) + 1), psnr_values, color='salmon')
    plt.xlabel('Image Index')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR for DWT Only Transformation')
    plt.savefig("dwt_only_psnr.png")  # Save the plot as an image file
    print("DWT Only PSNR plot saved as 'dwt_only_psnr.png'.")

# Main function to load images and calculate coefficient differences and PSNR for DWT Only
def main_dwt_only():
    data_directory = "./Medical-imaging-dataset"
    original_images = load_images(data_directory, target_size=(256, 256), batch_size=5)

    # Calculate, tabulate, and plot coefficient differences and PSNR for "DWT Only"
    calculate_dwt_only_coefficient_differences_and_psnr(original_images)

if __name__ == "__main__":
    main_dwt_only()
