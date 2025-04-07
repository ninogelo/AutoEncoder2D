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

# Function to compare original, DWT-only, and reconstructed images
def compare_images(original_images, dwt_images, reconstructed_images, num_images=5):
    plt.figure(figsize=(18, 15))
    for i in range(num_images):
        # Original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(original_images[i])
        plt.title("Original")
        plt.axis("off")

        # DWT LL component of original image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        LL_original = dwt_images[i, :, :, :]
        plt.imshow(LL_original[:, :, 0], cmap='gray')
        plt.title("DWT LL Original")
        plt.axis("off")

        # Reconstructed image
        ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(reconstructed_images[i])
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("comparison_results.png")  # Save the comparison as an image file
    print("Comparison plot saved as 'comparison_results.png'.")

# Function to calculate, tabulate, and plot coefficient differences
def calculate_coefficient_differences(original_images, dwt_images, reconstructed_images):
    # Apply DWT preprocessing to original and reconstructed images to match shapes
    original_dwt = preprocess_images_with_dwt(original_images, level=1)
    reconstructed_dwt = preprocess_images_with_dwt(reconstructed_images, level=1)

    differences_dwt_vs_reconstructed = []
    differences_original_vs_dwt = []
    differences_original_vs_reconstructed = []

    for i in range(len(original_dwt)):
        # Calculate differences between DWT-only and reconstructed
        diff_dwt_reconstructed = np.abs(dwt_images[i] - reconstructed_dwt[i])
        differences_dwt_vs_reconstructed.append(np.mean(diff_dwt_reconstructed))

        # Calculate differences between original and DWT-only
        diff_original_dwt = np.abs(original_dwt[i] - dwt_images[i])
        differences_original_vs_dwt.append(np.mean(diff_original_dwt))

        # Calculate differences between original and reconstructed
        diff_original_reconstructed = np.abs(original_dwt[i] - reconstructed_dwt[i])
        differences_original_vs_reconstructed.append(np.mean(diff_original_reconstructed))

    # Print summary of coefficient differences for each image
    for i in range(len(differences_dwt_vs_reconstructed)):
        print(f"Image {i+1}:")
        print(f"  Mean coefficient difference (DWT vs Reconstructed) = {differences_dwt_vs_reconstructed[i]}")
        print(f"  Mean coefficient difference (Original vs DWT) = {differences_original_vs_dwt[i]}")
        print(f"  Mean coefficient difference (Original vs Reconstructed) = {differences_original_vs_reconstructed[i]}")

    # Plot the mean coefficient differences
    plt.figure(figsize=(15, 8))
    indices = range(1, len(differences_dwt_vs_reconstructed) + 1)

    plt.bar(indices, differences_dwt_vs_reconstructed, color='skyblue', label='DWT vs Reconstructed')
    plt.bar(indices, differences_original_vs_dwt, color='lightgreen', label='Original vs DWT', alpha=0.7)
    plt.bar(indices, differences_original_vs_reconstructed, color='salmon', label='Original vs Reconstructed', alpha=0.7)

    plt.xlabel('Image Index')
    plt.ylabel('Mean Coefficient Difference')
    plt.title('Mean Coefficient Differences Between Original, DWT, and Reconstructed Images')
    plt.legend()
    plt.savefig("coefficient_differences.png")  # Save the plot as an image file
    print("Coefficient differences plot saved as 'coefficient_differences.png'.")

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

    # Compare the original, DWT-only, and reconstructed images
    compare_images(original_images, dwt_images, reconstructed_images, num_images=5)

    # Calculate, tabulate, and plot coefficient differences
    calculate_coefficient_differences(original_images, dwt_images, reconstructed_images)

if __name__ == "__main__":
    main()
