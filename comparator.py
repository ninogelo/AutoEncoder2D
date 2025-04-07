import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import pywt
import plaidml.keras
plaidml.keras.install_backend()  # Install PlaidML as the Keras backend
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the trained autoencoder model
autoencoder = load_model('autoencoder_model.h5')

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

# Function to calculate the mean squared error (MSE) between images
def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# Function to plot the original, reconstructed, and difference images
def reconstruct_images(autoencoder, generator):
    # Get a batch of images from the generator
    images = next(generator)  # Get the next image batch
    if images.shape[0] == 1:
        images = images[0]  # Remove batch dimension if there's only one image

    # Use the autoencoder to reconstruct the image
    reconstructed_images = autoencoder.predict(np.expand_dims(images, axis=0))[0]

    # Calculate the difference and MSE
    difference_image = np.abs(images - reconstructed_images)
    mse_value = calculate_mse(images, reconstructed_images)

    # Plot the images
    plt.figure(figsize=(18, 8))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(images)
    plt.title("Original")
    plt.axis('off')

    # Reconstructed image
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_images)
    plt.title("Reconstructed")
    plt.axis('off')

    # Difference image (heatmap)
    plt.subplot(1, 3, 3)
    plt.imshow(difference_image, cmap='hot', interpolation='nearest')
    plt.title("Difference (Heatmap)")
    plt.axis('off')

    # Save the plot
    plt.savefig("reconstructed_images_with_difference.png")
    print(f"Reconstructed images and differences saved as reconstructed_images_with_difference.png with MSE: {mse_value:.4f}")

# Main function to run the reconstruction
def main():
    data_directory = "./chest_xray_pneumonia/chest_xray/test"  # Path to your test images
    generator = load_and_preprocess_images(data_directory)
    reconstruct_images(autoencoder, generator)

if __name__ == "__main__":
    main()
