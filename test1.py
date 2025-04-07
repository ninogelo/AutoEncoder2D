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
    # Visualize each DWT component separately
    dwt_image = cA
    return dwt_image, coeffs2

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

# Function to plot images and heatmaps
def reconstruct_images(autoencoder, generator):
    images = next(generator)  # Get the next image batch
    images = images[0]  # Remove batch dimension

    # Debug: Print the max and min values of the original image
    print("Original image max value:", np.max(images))
    print("Original image min value:", np.min(images))

    # Use the autoencoder to reconstruct the image
    reconstructed_image = autoencoder.predict(np.expand_dims(images, axis=0))[0]

    # Debug: Print the max and min values of the autoencoder output
    print("Autoencoder output max value:", np.max(reconstructed_image))
    print("Autoencoder output min value:", np.min(reconstructed_image))

    # Perform 2D DWT
    dwt_image, coeffs = perform_2d_dwt(images[:, :, 0])  # Use the first channel for DWT

    # Normalize the DWT image
    dwt_image_normalized = dwt_image / np.max(dwt_image)

    # Debug: Print the max and min values of the DWT image
    print("DWT image max value:", np.max(dwt_image))
    print("DWT image min value:", np.min(dwt_image))

    # Normalize images to avoid scaling discrepancies
    images_normalized = images / np.max(images)
    reconstructed_image_normalized = reconstructed_image / np.max(reconstructed_image)

    # Ensure the autoencoder output is in the same range as the input
    reconstructed_image_normalized = np.clip(reconstructed_image_normalized, 0, 1)

    # Debug: Print the max and min values of the normalized autoencoder output
    print("Normalized autoencoder output max value:", np.max(reconstructed_image_normalized))
    print("Normalized autoencoder output min value:", np.min(reconstructed_image_normalized))

    # Resize dwt_image_normalized to match the original image size
    from PIL import Image
    dwt_image_resized = Image.fromarray((dwt_image_normalized * 255).astype(np.uint8))
    dwt_image_resized = dwt_image_resized.resize((256, 256), Image.LANCZOS)
    dwt_image_resized = np.array(dwt_image_resized) / 255.0

    # Convert dwt_image_resized to 3 channels
    dwt_image_resized = np.stack((dwt_image_resized,) * 3, axis=-1)

    # Debug: Print the max and min values of the resized DWT image
    print("Resized DWT image max value:", np.max(dwt_image_resized))
    print("Resized DWT image min value:", np.min(dwt_image_resized))

    # Calculate differences for heatmaps
    difference_original_reconstructed = np.abs(images_normalized - reconstructed_image_normalized)  # Difference for Original vs Reconstructed
    difference_original_dwt = np.abs(images_normalized - dwt_image_resized)  # Difference for Original vs DWT
    difference_dwt_reconstructed = np.abs(dwt_image_resized - reconstructed_image_normalized)  # Difference for DWT vs Reconstructed

    # Plot the images and heatmaps
    plt.figure(figsize=(24, 12))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(images)
    plt.title("Original Image")
    plt.axis('off')

    # Reconstructed Image
    plt.subplot(2, 3, 2)
    plt.imshow(reconstructed_image_normalized)
    plt.title("Reconstructed Image")
    plt.axis('off')

    # DWT Image
    plt.subplot(2, 3, 3)
    plt.imshow(dwt_image_resized, cmap='gray')
    plt.title("DWT Image")
    plt.axis('off')

    # Heatmap for Original vs Reconstructed difference
    plt.subplot(2, 3, 4)
    plt.imshow(difference_original_reconstructed, cmap='hot', interpolation='nearest')
    plt.title("Difference: Original vs Reconstructed")
    plt.colorbar()
    plt.axis('off')

    # Heatmap for Original vs DWT difference
    plt.subplot(2, 3, 5)
    plt.imshow(difference_original_dwt, cmap='hot', interpolation='nearest')
    plt.title("Difference: Original vs DWT")
    plt.colorbar()
    plt.axis('off')

    # Heatmap for DWT vs Reconstructed difference
    plt.subplot(2, 3, 6)
    plt.imshow(difference_dwt_reconstructed, cmap='hot', interpolation='nearest')
    plt.title("Difference: DWT vs Reconstructed")
    plt.colorbar()
    plt.axis('off')

    # Save the plot
    plt.savefig("test1.png")
    print("Images and heatmaps saved as test1.png")

# Main function to run the reconstruction
def main():
    data_directory = "./chest_xray_pneumonia/chest_xray/test"  # Path to your test images
    generator = load_and_preprocess_images(data_directory)
    reconstruct_images(autoencoder, generator)

if __name__ == "__main__":
    main()