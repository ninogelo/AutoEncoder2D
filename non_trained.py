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

# Helper function to preprocess images with DWT
def preprocess_images_with_dwt(image_batch):
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

# Function to calculate and plot coefficient differences between original and DWT-transformed images
def calculate_dwt_coefficient_differences(original_images):
    original_dwt = preprocess_images_with_dwt(original_images)

    differences = []
    for i in range(len(original_dwt)):
        # Calculate the difference between the original image and the DWT LL component
        original_image = original_images[i, :, :, :]
        LL = original_dwt[i, :, :, :]

        # Resize original to match LL size
        original_resized = original_image[::2, ::2, :]

        diff = np.abs(original_resized - LL)
        differences.append(np.mean(diff))

    # Print summary of coefficient differences for each image
    for i, diff in enumerate(differences):
        print(f"Image {i+1}: Mean coefficient difference = {diff}")

    # Plot the mean coefficient differences
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(differences) + 1), differences, color='skyblue')
    plt.xlabel('Image Index')
    plt.ylabel('Mean Coefficient Difference')
    plt.title('Mean Coefficient Differences Between Original and DWT-Transformed Images')
    plt.savefig("dwt_coefficient_differences.png")  # Save the plot as an image file
    print("Coefficient differences plot saved as 'dwt_coefficient_differences.png'.")

# Main function to load images and calculate coefficient differences
def main():
    data_directory = "./Medical-imaging-dataset"
    original_images = load_images(data_directory, target_size=(256, 256), batch_size=5)

    # Calculate, tabulate, and plot coefficient differences for DWT
    calculate_dwt_coefficient_differences(original_images)

if __name__ == "__main__":
    main()
