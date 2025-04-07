import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


import numpy as np
import pywt
import plaidml.keras
plaidml.keras.install_backend()  # Use PlaidML as the Keras backend
import keras
from keras.models import load_model
import cv2  # Import OpenCV for image resizing
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# Load the trained autoencoder model
autoencoder = load_model('autoencoder_model.h5')

# Function to pad the image to make dimensions even and divisible by 2^level
def pad_for_swt(image, level=2):
    height, width = image.shape
    pad_height = (2**level - height % (2**level)) % (2**level)
    pad_width = (2**level - width % (2**level)) % (2**level)
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='reflect')
    return padded_image, (height, width)

# Function to crop the image back to its original size
def crop_to_original(image, original_shape):
    return image[:original_shape[0], :original_shape[1]]

# Function to apply multi-level 2D SWT to a grayscale image
def apply_swt(image, level=2, wavelet='haar'):
    coeffs = pywt.swt2(image, wavelet, level=level)
    return coeffs

# Function to apply inverse SWT to a grayscale image
def apply_iswt(coeffs, wavelet='haar'):
    reconstructed_image = pywt.iswt2(coeffs, wavelet)
    return np.clip(reconstructed_image, 0, 1)  # Clip values to be in the range [0, 1]

# Function to enhance SWT coefficients using the autoencoder
def enhance_with_autoencoder(coeffs):
    enhanced_coeffs = []
    for level_coeffs in coeffs:
        cA, (cH, cV, cD) = level_coeffs
        # Resize cA to (256, 256) for the autoencoder
        cA_resized = cv2.resize(cA, (256, 256))

        # Add batch and channel dimensions for the autoencoder
        cA_resized = np.expand_dims(cA_resized, axis=0)
        cA_resized = np.expand_dims(cA_resized, axis=-1)

        # Use the autoencoder to enhance coefficients
        enhanced_cA = autoencoder.predict(cA_resized)

        # Convert back to the original shape
        enhanced_cA = cv2.resize(enhanced_cA[0, :, :, 0], (cA.shape[1], cA.shape[0]))

        # Return the enhanced approximation coefficients with the original detail coefficients
        enhanced_coeffs.append((enhanced_cA, (cH, cV, cD)))
    return enhanced_coeffs

# Function to quantize coefficients
def quantize(coeffs, step_size=10):
    quantized_coeffs = []
    for level_coeffs in coeffs:
        cA, (cH, cV, cD) = level_coeffs
        quantized_cA = np.round(cA / step_size) * step_size
        quantized_cH = np.round(cH / step_size) * step_size
        quantized_cV = np.round(cV / step_size) * step_size
        quantized_cD = np.round(cD / step_size) * step_size
        quantized_coeffs.append((quantized_cA, (quantized_cH, quantized_cV, quantized_cD)))
    return quantized_coeffs

# Function to calculate the size of compressed coefficients
def calculate_compressed_size(coeffs):
    total_size = 0
    for level_coeffs in coeffs:
        cA, (cH, cV, cD) = level_coeffs
        total_size += cA.size + cH.size + cV.size + cD.size
    return total_size

# Function to calculate the compression ratio
def calculate_compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

# Main function for compression and reconstruction
def main(image, step_size, wavelet='haar', level=2):
    # Convert image to numpy array and normalize to [0, 1]
    image = np.array(image) / 255.0
    if len(image.shape) == 3:  # If the image is RGB, convert to grayscale
        image = np.mean(image, axis=2)

    # Pad the image to make dimensions divisible by 2^level
    padded_image, original_shape = pad_for_swt(image, level=level)
    original_size = image.size

    # Apply multi-level 2D SWT
    coeffs = apply_swt(padded_image, level=level, wavelet=wavelet)
    compressed_size = calculate_compressed_size(coeffs)
    compression_ratio = calculate_compression_ratio(original_size, compressed_size)
    st.write(f"Compression Ratio (before autoencoder): {compression_ratio:.2f}")

    # Enhance coefficients using the autoencoder
    enhanced_coeffs = enhance_with_autoencoder(coeffs)

    # Quantize coefficients
    quantized_coeffs = quantize(enhanced_coeffs, step_size)

    # Reconstruct the image using inverse SWT
    reconstructed_image = apply_iswt(quantized_coeffs, wavelet=wavelet)

    # Crop the reconstructed image back to its original size
    reconstructed_image = crop_to_original(reconstructed_image, original_shape)

    # Plot the original and reconstructed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("Reconstructed Image with Autoencoder")
    plt.axis('off')

    # Save the plot as an image file
    plt.savefig("swt_reconstructed_comparison.png")
    st.image("swt_reconstructed_comparison.png")

# Streamlit UI
st.title("Image Compression and Reconstruction with SWT and Autoencoder")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
step_size = st.slider("Quantization Step Size", min_value=1, max_value=50, value=10)
wavelet = st.selectbox("Wavelet", ["haar", "db1", "db2", "sym2"])  # Add more wavelets if needed
level = st.slider("Decomposition Level", min_value=1, max_value=4, value=2)  # Adjust max_value based on image size

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Processing...")
    main(image, step_size, wavelet, level)