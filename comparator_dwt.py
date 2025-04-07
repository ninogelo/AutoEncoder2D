import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import pywt
import plaidml.keras
plaidml.keras.install_backend()  # Use PlaidML as the Keras backend
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt

# Load the trained autoencoder model
autoencoder = load_model('autoencoder_model.h5')

# Function to perform 2D DWT
def perform_2d_dwt(image):
    coeffs2 = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs2
    dwt_image = np.hstack((cA, cH, cV, cD))  # Combine coefficients for visualization
    return dwt_image, coeffs2

# Function to reconstruct the image from DWT coefficients
def reconstruct_from_dwt(coeffs):
    return pywt.idwt2(coeffs, 'haar')

# Function to preprocess and load images for reconstruction
def load_and_preprocess_images(data_directory):
    data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    generator = data_gen.flow_from_directory(
        data_directory,
        target_size=(256, 256),
        batch_size=1,  # Load one image at a time for reconstruction
        class_mode=None,
        shuffle=False  # Keep images in order
    )
    return generator

# Function to calculate compression ratio
def calculate_compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

# Function to plot images and heatmaps
def reconstruct_images(autoencoder, generator):
    images = next(generator)  # Get the next image batch
    images = images[0]  # Remove batch dimension

    # Perform 2D DWT
    dwt_image, coeffs = perform_2d_dwt(images[:, :, 0])  # Use the first channel for DWT

    # Reconstruct the image from DWT coefficients
    reconstructed_dwt_image = reconstruct_from_dwt(coeffs)

    # Use the autoencoder to further reconstruct the image
    images_expanded = np.expand_dims(images, axis=0)  # Add batch dimension
    reconstructed_images = autoencoder.predict(images_expanded)[0]

    # Calculate the differences for heatmaps
    difference_dwt = np.abs(images[:, :, 0] - reconstructed_dwt_image)  # Difference for DWT
    difference_reconstructed = np.abs(images - reconstructed_images)  # Difference for Autoencoder
    heatmap_dwt = difference_dwt  # Heatmap for DWT
    heatmap_reconstructed = np.mean(difference_reconstructed, axis=2)  # Heatmap for Autoencoder

    # Calculate compression ratio
    original_size = images.size
    compressed_size = sum(c.size for c in [coeffs[0], *coeffs[1]])  # Fixed line
    compression_ratio = calculate_compression_ratio(original_size, compressed_size)

    # Plot the images and heatmaps
    plt.figure(figsize=(20, 8))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(images)
    plt.title("Original Image")
    plt.axis('off')

    # DWT image
    plt.subplot(2, 3, 2)
    plt.imshow(dwt_image, cmap='gray')
    plt.title("DWT Image")
    plt.axis('off')

    # Reconstructed image
    plt.subplot(2, 3, 3)
    plt.imshow(reconstructed_images)
    plt.title("Reconstructed Image")
    plt.axis('off')

    # Heatmap for DWT difference
    plt.subplot(2, 3, 4)
    plt.imshow(heatmap_dwt, cmap='hot', interpolation='nearest')
    plt.title("DWT Difference Heatmap")
    plt.colorbar()
    plt.axis('off')

    # Heatmap for Autoencoder difference
    plt.subplot(2, 3, 5)
    plt.imshow(heatmap_reconstructed, cmap='hot', interpolation='nearest')
    plt.title("Reconstructed Difference Heatmap")
    plt.colorbar()
    plt.axis('off')

    # Display compression ratio
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.5, f"Compression Ratio: {compression_ratio:.2f}", fontsize=12)
    plt.axis('off')

    # Save the plot
    plt.savefig("dwt_reconstructed_comparison.png")
    print("Images and heatmaps saved as dwt_reconstructed_comparison.png")


# Main function to run the reconstruction
def main():
    data_directory = "./chest_xray_pneumonia/chest_xray/test"  # Path to your test images
    generator = load_and_preprocess_images(data_directory)
    reconstruct_images(autoencoder, generator)

if __name__ == "__main__":
    main()
