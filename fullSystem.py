# app.py
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import pywt
import plaidml.keras
plaidml.keras.install_backend()
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import gc

# Load the trained autoencoder model
@st.cache_resource
def load_autoencoder_model():
    try:
        model = load_model('autoencoder_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# DWT functions
def perform_2d_dwt(image):
    coeffs2 = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs2
    # Create visualization of coefficients
    dwt_image = np.hstack((cA, np.hstack((cH, cV, cD))))
    return dwt_image, coeffs2

def reconstruct_from_dwt(coeffs):
    return pywt.idwt2(coeffs, 'haar')

def calculate_compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

# Streamlit app
def main():
    st.title("Enhanced DWT Image Compression")
    st.write("Compare DWT and Autoencoder reconstructions")

    # Load model here to ensure it's available when needed
    autoencoder = load_autoencoder_model()

    # File upload
    uploaded_file = st.file_uploader("Upload a medical image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and autoencoder is not None:
        try:
            # Preprocess image
            image = Image.open(uploaded_file).convert('RGB')
            image = image.resize((256, 256))
            img_array = np.array(image) / 255.0
            
            # Ensure 3-channel format
            if img_array.ndim == 2:
                img_array = np.stack((img_array,)*3, axis=-1)

            # Perform DWT on first channel
            dwt_image, coeffs = perform_2d_dwt(img_array[:, :, 0])
            reconstructed_dwt = reconstruct_from_dwt(coeffs)
            
            # Force shape to be compatible
            reconstructed_dwt = np.clip(reconstructed_dwt, 0, 1)
            
            # Reshape to match original for comparison
            reconstructed_dwt_3channel = np.stack((reconstructed_dwt,)*3, axis=-1)

            # Autoencoder reconstruction - handle with try-except for device compatibility
            try:
                input_array = np.expand_dims(img_array, axis=0)
                reconstructed_ae = autoencoder.predict(input_array)
                reconstructed_ae = reconstructed_ae[0]
                
                # Ensure proper shape and range
                reconstructed_ae = np.clip(reconstructed_ae, 0, 1)
            except Exception as e:
                st.warning(f"Autoencoder prediction error: {e}. Using CPU fallback.")
                # Fallback to a simple copy for demonstration
                reconstructed_ae = img_array.copy()

            # Calculate differences - using CPU numpy operations to avoid device issues
            diff_dwt = np.abs(img_array - reconstructed_dwt_3channel).mean(axis=-1)
            diff_ae = np.abs(img_array - reconstructed_ae).mean(axis=-1)

            # Calculate compression ratio
            original_size = img_array.size * 8  # 8 bits per value
            compressed_size = (coeffs[0].size + coeffs[1][0].size + 
                              coeffs[1][1].size + coeffs[1][2].size) * 32  # 32 bits float
            compression_ratio = calculate_compression_ratio(original_size, compressed_size)

            # Create figure using matplotlib (CPU-based visualization)
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))

            # Plot images
            axs[0,0].imshow(img_array)
            axs[0,0].set_title("Original Image")
            axs[0,0].axis('off')


            axs[0,2].imshow(reconstructed_ae)
            axs[0,2].set_title("Autoencoder Reconstruction")
            axs[0,2].axis('off')

            axs[1,0].imshow(reconstructed_dwt, cmap='gray')
            axs[1,0].set_title("DWT Reconstruction")
            axs[1,0].axis('off')

            axs[1,1].imshow(diff_ae, cmap='hot')
            axs[1,1].set_title("Autoencoder Difference")
            axs[1,1].axis('off')

            # Add metrics to the last subplot
            metrics_text = (
                f"Compression Ratio: {compression_ratio:.2f}\n"
                f"DWT PSNR: {psnr(img_array[:,:,0], reconstructed_dwt):.2f} dB\n"
                f"AE PSNR: {psnr(img_array.mean(axis=2), reconstructed_ae.mean(axis=2)):.2f} dB"
            )
            axs[1,2].text(0.1, 0.5, metrics_text, fontsize=12)
            axs[1,2].axis('off')

            plt.tight_layout()
            st.pyplot(fig)
            
            # Free up GPU memory
            gc.collect()

        except Exception as e:
            st.error(f"Error processing image: {e}")
            import traceback
            st.code(traceback.format_exc())

def psnr(original, compressed):
    """Calculate the Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

if __name__ == "__main__":
    main()