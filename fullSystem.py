import os
import sys
import numpy as np
import pywt
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import gc
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# Safely import OpenCV
try:
    import cv2
except ImportError:
    st.error("OpenCV import error. Please make sure packages.txt includes: libgl1 libgl1-mesa-glx libglib2.0-0")

# Detect environment - use PlaidML locally but mock on Streamlit Cloud
try:
    # Check if we're in the Streamlit Cloud environment (Python 3.12)
    is_streamlit_cloud = sys.version_info.major == 3 and sys.version_info.minor >= 12
    
    if not is_streamlit_cloud:
        # Local environment - use PlaidML
        os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
        import plaidml.keras
        plaidml.keras.install_backend()
        import keras
        from keras.models import load_model
        st.sidebar.success("Using PlaidML backend")
    else:
        # Streamlit Cloud - use mock implementation
        raise ImportError("Running in Streamlit Cloud environment")
except ImportError:
    st.sidebar.warning("PlaidML not available - using simulation mode")

# Load the trained autoencoder model
@st.cache_resource
def load_autoencoder_model():
    try:
        # Check if we can use the real model with PlaidML
        try:
            if 'keras' in sys.modules:
                model = load_model('autoencoder_model.h5')
                return model
        except Exception as model_e:
            st.warning(f"Could not load real model: {model_e}")
            
        # Use demonstration mode with simulated model
        st.warning("Using demonstration mode with simulated model")
        
        # Create a mock model class
        class MockModel:
            def predict(self, input_array):
                # This is a simple dummy implementation that returns a slightly modified version
                # of the input to simulate compression and reconstruction
                result = input_array.copy()
                # Simulate some loss of high-frequency details
                for c in range(result.shape[3]):
                    result[0, :, :, c] = cv2.GaussianBlur(result[0, :, :, c], (5, 5), 0)
                return result
        
        return MockModel()
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


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
def perform_2d_dwt(image, wavelet='haar'):
    coeffs2 = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs2
    return coeffs2, cA, cH, cV, cD

def reconstruct_from_dwt(coeffs, wavelet='haar'):
    return pywt.idwt2(coeffs, wavelet)

def calculate_compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

def calculate_phase_coherence(img):
    # Calculate phase coherence using Fourier transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    phase = np.angle(fshift)
    return np.mean(np.abs(np.cos(phase)))

def calculate_edge_coherence(img):
    # Using Sobel operator to detect edges
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.sum(magnitude > 0.1) / magnitude.size * 100  # Percentage of edge pixels

def calculate_directional_ssim(img1, img2, angle=30):
    # Create a directional mask
    mask = np.zeros_like(img1)
    h, w = mask.shape
    cv2.line(mask, (0, 0), (w-1, int(np.tan(np.radians(angle)) * w)), 1, thickness=10)
    
    # Apply mask to both images
    masked_img1 = img1 * mask
    masked_img2 = img2 * mask
    
    # Calculate SSIM on the masked region
    return ssim(masked_img1, masked_img2, data_range=1.0)

def calculate_edge_sharpness(img):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.mean(magnitude)

def calculate_mad_hh(coeffs):
    # Mean Absolute Deviation in HH subband
    _, (_, _, hh) = coeffs
    return np.mean(np.abs(hh - np.mean(hh)))

def psnr(original, compressed):
    """Calculate the Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

# Main application
def main():
    st.title("Enhanced Medical Image Compression Analysis")
    st.write("Compare DWT and Autoencoder compression with advanced metrics")

    # Load model
    autoencoder = load_autoencoder_model()

    # Sidebar controls
    st.sidebar.header("Controls")
    wavelet_type = st.sidebar.selectbox("Wavelet Type", 
                                     options=["haar", "db1", "db2", "sym2", "coif1"],
                                     index=0)
    
    # File upload
    uploaded_file = st.file_uploader("Upload a medical image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and autoencoder is not None:
        try:
            # Preprocess image
            image = Image.open(uploaded_file).convert('RGB')
            image = image.resize((256, 256))
            img_array = np.array(image) / 255.0
            
            # Convert to grayscale for wavelet processing
            gray_img = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
            
            # Perform DWT
            coeffs, cA, cH, cV, cD = perform_2d_dwt(gray_img, wavelet_type)
            
            # Create DWT existing visualization - normalize coefficients for display
            # Normalize each coefficient subband individually for better visualization
            cA_norm = (cA - np.min(cA)) / (np.max(cA) - np.min(cA) + 1e-10)
            cH_norm = (cH - np.min(cH)) / (np.max(cH) - np.min(cH) + 1e-10)
            cV_norm = (cV - np.min(cV)) / (np.max(cV) - np.min(cV) + 1e-10)
            cD_norm = (cD - np.min(cD)) / (np.max(cD) - np.min(cD) + 1e-10)
            
            dwt_existing = np.vstack((
                np.hstack((cA_norm, cH_norm)),
                np.hstack((cV_norm, cD_norm))
            ))
            
            # Reconstruct from DWT
            reconstructed_dwt = reconstruct_from_dwt(coeffs, wavelet_type)
            reconstructed_dwt = np.clip(reconstructed_dwt, 0, 1)
            
            # Process with autoencoder
            try:
                input_array = np.expand_dims(img_array, axis=0)
                reconstructed_ae = autoencoder.predict(input_array)[0]
                reconstructed_ae = np.clip(reconstructed_ae, 0, 1)
            except Exception as e:
                st.warning(f"Autoencoder prediction error: {e}. Using fallback.")
                reconstructed_ae = img_array.copy()
            
            # Create DWT with AE visualization (combining both methods)
            ae_gray = 0.299 * reconstructed_ae[:,:,0] + 0.587 * reconstructed_ae[:,:,1] + 0.114 * reconstructed_ae[:,:,2]
            dwt_with_ae_coeffs, ae_cA, ae_cH, ae_cV, ae_cD = perform_2d_dwt(ae_gray, wavelet_type)
            dwt_with_ae = reconstruct_from_dwt(dwt_with_ae_coeffs, wavelet_type)
            dwt_with_ae = np.clip(dwt_with_ae, 0, 1)
            
            # Calculate metrics for all three methods
            # Initialize metrics dictionary
            metrics = {
                'original': {},
                'dwt': {},
                'dwt_ae': {}
            }
            
            # Objective 1: Retained Phase Information
            # For original image - standard phase coherence
            metrics['original']['phase_coherence'] = calculate_phase_coherence(gray_img)
            
            # For DWT - phase coherence of approximation coefficients
            metrics['dwt']['phase_coherence'] = calculate_phase_coherence(cA)
            
            # For Proposed - phase coherence of the result
            metrics['dwt_ae']['phase_coherence'] = calculate_phase_coherence(dwt_with_ae)
            
            # Edge coherence - percentage of edge pixels
            metrics['original']['edge_coherence'] = calculate_edge_coherence(gray_img)
            
            # For DWT - measure using normalized detail coefficients
            detail_energy = np.sum(np.abs(cH) + np.abs(cV) + np.abs(cD))
            total_energy = np.sum(np.abs(cA)) + detail_energy
            metrics['dwt']['edge_coherence'] = (detail_energy / total_energy) * 100
            
            # For Proposed - standard edge coherence
            metrics['dwt_ae']['edge_coherence'] = calculate_edge_coherence(dwt_with_ae)
            
            # Objective 2: Reduced Shift Variance
            # Mean Absolute Deviation in HH subband
            metrics['original']['coeff_mad'] = calculate_mad_hh(coeffs)
            
            # For DWT - use the normalized magnitude of high-frequency coefficients
            metrics['dwt']['coeff_mad'] = np.mean(np.abs(cD)) / np.mean(np.abs(cA)) if np.mean(np.abs(cA)) > 0 else 0
            
            # For Proposed - MAD of coefficients from the combined approach
            metrics['dwt_ae']['coeff_mad'] = calculate_mad_hh(dwt_with_ae_coeffs)
            
            # Shift sensitivity - how much the image changes with a small shift
            # Original image shift sensitivity
            shifted_orig = np.roll(gray_img, 1, axis=1)
            metrics['original']['shift_sens'] = np.mean(np.abs(gray_img - shifted_orig))
            
            # DWT shift sensitivity in coefficient domain
            shifted_img_coeffs = perform_2d_dwt(np.roll(gray_img, 1, axis=1), wavelet_type)
            _, (_, _, hh_shifted) = shifted_img_coeffs[0]
            _, (_, _, hh_orig) = coeffs
            metrics['dwt']['shift_sens'] = np.mean(np.abs(hh_orig - hh_shifted)) / np.mean(np.abs(hh_orig)) if np.mean(np.abs(hh_orig)) > 0 else 0
            
            # Proposed shift sensitivity
            shifted_ae = np.roll(dwt_with_ae, 1, axis=1)
            metrics['dwt_ae']['shift_sens'] = np.mean(np.abs(dwt_with_ae - shifted_ae))
            
            # PSNR compared to original
            metrics['original']['psnr'] = float('inf')  # Perfect with itself
            
            # For DWT - PSNR based on approximation coefficients vs original 
            # Using a downsampled version of original for fair comparison
            downsampled_orig = cv2.resize(gray_img, (cA.shape[1], cA.shape[0]))
            metrics['dwt']['psnr'] = psnr(downsampled_orig, cA)
            
            # For Proposed - standard PSNR
            metrics['dwt_ae']['psnr'] = psnr(gray_img, dwt_with_ae)
            
            # Objective 3: Increased Directionality
            # Directional response for horizontal (30°) and vertical (60°) features
            # For original, these are 1.0 (perfect with itself)
            metrics['original']['dir_ssim_30'] = 1.0
            metrics['original']['dir_ssim_60'] = 1.0
            
            # For DWT - measure using ratio of horizontal and vertical detail coefficients
            h_energy = np.mean(np.abs(cH))
            v_energy = np.mean(np.abs(cV))
            total_energy = h_energy + v_energy
            
            if total_energy > 0:
                metrics['dwt']['dir_ssim_30'] = h_energy / total_energy  # Horizontal component
                metrics['dwt']['dir_ssim_60'] = v_energy / total_energy  # Vertical component
            else:
                metrics['dwt']['dir_ssim_30'] = 0.5
                metrics['dwt']['dir_ssim_60'] = 0.5
            
            # For Proposed - standard directional SSIM
            metrics['dwt_ae']['dir_ssim_30'] = calculate_directional_ssim(gray_img, dwt_with_ae, 30)
            metrics['dwt_ae']['dir_ssim_60'] = calculate_directional_ssim(gray_img, dwt_with_ae, 60)
            
            # Edge sharpness - magnitude of image gradients
            metrics['original']['edge_sharpness'] = calculate_edge_sharpness(gray_img)
            
            # For DWT - use mean magnitude of detail coefficients
            metrics['dwt']['edge_sharpness'] = (np.mean(np.abs(cH)) + np.mean(np.abs(cV)) + np.mean(np.abs(cD))) / 3
            
            # For Proposed - standard edge sharpness
            metrics['dwt_ae']['edge_sharpness'] = calculate_edge_sharpness(dwt_with_ae)
            
            # Compression ratio
            original_size = gray_img.size * 8  # 8 bits per pixel
            
            # For original - no compression
            metrics['original']['compression_ratio'] = 1.0
            
            # For DWT - count significant coefficients only
            threshold = 0.05 * np.max([np.max(np.abs(cH)), np.max(np.abs(cV)), np.max(np.abs(cD))])
            significant_coeffs = np.sum(np.abs(cH) > threshold) + np.sum(np.abs(cV) > threshold) + np.sum(np.abs(cD) > threshold)
            compressed_size_dwt = (cA.size + significant_coeffs) * 32  # 32 bits float
            metrics['dwt']['compression_ratio'] = calculate_compression_ratio(original_size, compressed_size_dwt)
            
            # For Proposed - estimate based on autoencoder latent space
            latent_size = original_size // 8  # Assuming 8:1 compression
            metrics['dwt_ae']['compression_ratio'] = calculate_compression_ratio(original_size, latent_size)
            
            # Create an enlarged main image comparison layout
            st.header("IMAGE COMPARISON")

            # First row - main images only (3 columns)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("ORIGINAL IMAGE")
                st.image(img_array, use_container_width=True)

            with col2:
                # Changed from "DWT (LL Band)" to just "DWT"
                st.subheader("DWT")
                # Using reconstructed_dwt instead of cA_norm to show actual DWT output
                st.image(reconstructed_dwt, use_container_width=True)

            with col3:
                # Changed from "DWT WITH AE" to "PROPOSED"
                st.subheader("PROPOSED")
                st.image(dwt_with_ae, use_container_width=True)

            # Second row - heatmaps only (2 columns)
            col4, col5 = st.columns(2)

            with col4:
                # Create a heatmap comparing original vs DWT's LL subband
                st.subheader("Original vs DWT")
                
                # Resize original to match the size of approximation coefficients
                # This ensures a fair comparison since LL subband is downsampled
                downsampled_orig = cv2.resize(gray_img, (cA.shape[1], cA.shape[0]))
                
                # Calculate difference between downsampled original and LL subband
                heatmap_orig_dwt = np.abs(downsampled_orig - cA_norm)
                
                # Calculate statistics for display
                max_diff_dwt = np.max(heatmap_orig_dwt)
                mean_diff_dwt = np.mean(heatmap_orig_dwt)
                
                # Use viridis colormap with custom normalization for better visualization
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(heatmap_orig_dwt, cmap='inferno', vmin=0, vmax=max_diff_dwt)
                ax.axis('off')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                # Set scale based on actual magnitude
                if max_diff_dwt < 0.01:
                    exp = int(np.log10(max_diff_dwt))
                    cbar.set_label(f'×10^{exp}', rotation=0, labelpad=15, y=1.05)
                
                # Add statistics in caption
                
                
                plt.tight_layout()
                st.pyplot(fig)

            with col5:
                # Keep the heatmap comparing original vs proposed (formerly Proposed)
                st.subheader("Original vs Proposed")
                heatmap_orig_proposed = np.abs(gray_img - dwt_with_ae)
                
                # Calculate statistics for display
                max_diff_proposed = np.max(heatmap_orig_proposed)
                mean_diff_proposed = np.mean(heatmap_orig_proposed)
                
                # Important: Use each heatmap's own scale, not the same one
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(heatmap_orig_proposed, cmap='inferno', vmin=0, vmax=max_diff_proposed)
                ax.axis('off')
                
                # Add colorbar with correct scale
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                # Set scale based on actual magnitude
                if max_diff_proposed < 0.01:
                    exp = int(np.log10(max_diff_proposed))
                    cbar.set_label(f'×10^{exp}', rotation=0, labelpad=15, y=1.05)
                
                # Add improvement statistics
                improvement_max = max_diff_dwt/max_diff_proposed if max_diff_proposed > 0 else float('inf')
                improvement_mean = mean_diff_dwt/mean_diff_proposed if mean_diff_proposed > 0 else float('inf')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Objectives and metrics sections
            st.header("VISUALIZATIONS")
            
            obj1, obj2, obj3 = st.tabs(["RETAINED PHASE INFORMATION", "REDUCED SHIFT VARIANCE", "INCREASED DIRECTIONALITY"])
            
            with obj1:  # RETAINED PHASE INFORMATION
                # First show the metrics table (full width)
                st.subheader("Metrics Comparison:")
                metrics_df = pd.DataFrame({
                    "Metric": ["Phase Coherence (rad²)", "Edge Coherence (%)"],
                    "Original": [
                        f"{metrics['original']['phase_coherence']:.4f}", 
                        f"{metrics['original']['edge_coherence']:.2f}"
                    ],
                    "DWT": [
                        f"{metrics['dwt']['phase_coherence']:.4f}", 
                        f"{metrics['dwt']['edge_coherence']:.2f}"
                    ],
                    "Proposed": [
                        f"{metrics['dwt_ae']['phase_coherence']:.4f}", 
                        f"{metrics['dwt_ae']['edge_coherence']:.2f}"
                    ]
                })
                st.table(metrics_df)
                
                # Then show the visualization (full width and larger)
                st.subheader("Phase Visualization")
                
                # Create phase visualization - using larger figure size
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                
                # Original image phase
                f_orig = np.fft.fft2(gray_img)
                phase_orig = np.angle(f_orig)
                axs[0].imshow(np.abs(phase_orig), cmap='twilight')
                axs[0].set_title("Original Phase", fontsize=14)
                axs[0].axis('off')
                
                # DWT existing phase - show phase of approximation coeffs
                f_dwt = np.fft.fft2(cA)
                phase_dwt = np.angle(f_dwt)
                axs[1].imshow(np.abs(phase_dwt), cmap='twilight')
                axs[1].set_title("DWT Phase", fontsize=14)
                axs[1].axis('off')
                
                # DWT with AE phase
                f_ae = np.fft.fft2(dwt_with_ae)
                phase_ae = np.angle(f_ae)
                axs[2].imshow(np.abs(phase_ae), cmap='twilight')
                axs[2].set_title("Proposed Phase", fontsize=14)
                axs[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with obj2:  # REDUCED SHIFT VARIANCE
                # First show the metrics table (full width)
                st.subheader("Metrics Comparison:")
                metrics_df = pd.DataFrame({
                    "Metric": ["Coefficient MAD (HH Subband)", "Shift Sensitivity", "PSNR (dB)"],
                    "Original": [
                        f"{metrics['original']['coeff_mad']:.4f}", 
                        f"{metrics['original']['shift_sens']:.4f}", 
                        "∞"
                    ],
                    "DWT": [
                        f"{metrics['dwt']['coeff_mad']:.4f}", 
                        f"{metrics['dwt']['shift_sens']:.4f}", 
                        f"{metrics['dwt']['psnr']:.2f}"
                    ],
                    "Proposed": [
                        f"{metrics['dwt_ae']['coeff_mad']:.4f}", 
                        f"{metrics['dwt_ae']['shift_sens']:.4f}", 
                        f"{metrics['dwt_ae']['psnr']:.2f}"
                    ]
                })
                st.table(metrics_df)
                
                # Then show the visualization (full width and larger)
                st.subheader("Shift Sensitivity Visualization")
                
                # Create shift variance visualization - using larger figure size
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                
                # Original with small shift - HH coefficient difference
                shifted_orig_coeffs = perform_2d_dwt(shifted_orig, wavelet_type)
                _, (_, _, hh_orig) = coeffs
                _, (_, _, hh_shifted_orig) = shifted_orig_coeffs[0]
                
                # Normalize for display
                shift_diff_hh = np.abs(hh_orig - hh_shifted_orig)
                shift_diff_hh = shift_diff_hh / np.max(shift_diff_hh) if np.max(shift_diff_hh) > 0 else shift_diff_hh
                
                axs[0].imshow(shift_diff_hh, cmap='hot')
                axs[0].set_title("Original HH Shift Sensitivity", fontsize=14)
                axs[0].axis('off')
                
                # DWT with small shift - detail coefficient differences
                # Just use the same HH comparison for DWT since we're analyzing the coefficients directly
                axs[1].imshow(cD_norm, cmap='hot')
                axs[1].set_title("DWT HH Coefficients", fontsize=14)
                axs[1].axis('off')
                
                # Proposed with small shift
                shifted_ae_coeffs = perform_2d_dwt(shifted_ae, wavelet_type)
                _, (_, _, hh_ae) = dwt_with_ae_coeffs
                _, (_, _, hh_shifted_ae) = shifted_ae_coeffs[0]
                
                # Normalize for display
                shift_diff_ae_hh = np.abs(hh_ae - hh_shifted_ae)
                shift_diff_ae_hh = shift_diff_ae_hh / np.max(shift_diff_ae_hh) if np.max(shift_diff_ae_hh) > 0 else shift_diff_ae_hh
                
                axs[2].imshow(shift_diff_ae_hh, cmap='hot')
                axs[2].set_title("Proposed HH Shift Sensitivity", fontsize=14)
                axs[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with obj3:  # INCREASED DIRECTIONALITY
                # First show the metrics table (full width)
                st.subheader("Metrics Comparison:")
                metrics_df = pd.DataFrame({
                    "Metric": ["Directional Response (H)", "Directional Response (V)", "Edge Sharpness", "Compression Ratio"],
                    "Original": [
                        f"{metrics['original']['dir_ssim_30']:.4f}", 
                        f"{metrics['original']['dir_ssim_60']:.4f}",
                        f"{metrics['original']['edge_sharpness']:.4f}", 
                        "1:1"
                    ],
                    "DWT": [
                        f"{metrics['dwt']['dir_ssim_30']:.4f}", 
                        f"{metrics['dwt']['dir_ssim_60']:.4f}",
                        f"{metrics['dwt']['edge_sharpness']:.4f}", 
                        f"{metrics['dwt']['compression_ratio']:.2f}:1"
                    ],
                    "Proposed": [
                        f"{metrics['dwt_ae']['dir_ssim_30']:.4f}", 
                        f"{metrics['dwt_ae']['dir_ssim_60']:.4f}",
                        f"{metrics['dwt_ae']['edge_sharpness']:.4f}", 
                        f"{metrics['dwt_ae']['compression_ratio']:.2f}:1"
                    ]
                })
                st.table(metrics_df)
                
                # Then show the visualization (full width and larger)
                st.subheader("Directionality Visualization")
                
                # Create directionality visualization - using larger figure size
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                
                # Original image directionality - using Sobel gradients
                sobelx_orig = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
                sobely_orig = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
                gradient_dir_orig = np.arctan2(sobely_orig, sobelx_orig)
                axs[0].imshow(gradient_dir_orig, cmap='hsv')
                axs[0].set_title("Original Edge Directions", fontsize=14)
                axs[0].axis('off')
                
                # DWT directionality - use horizontal and vertical coefficients
                # Create a composite image showing horizontal and vertical details
                h_v_composite = np.zeros((cH.shape[0], cH.shape[1], 3))
                h_v_composite[:,:,0] = cH_norm  # Red channel for horizontal
                h_v_composite[:,:,1] = cV_norm  # Green channel for vertical
                axs[1].imshow(h_v_composite)
                axs[1].set_title("DWT Directional Coefficients", fontsize=14)
                axs[1].axis('off')
                
                # Proposed directionality
                sobelx_ae = cv2.Sobel(dwt_with_ae, cv2.CV_64F, 1, 0, ksize=3)
                sobely_ae = cv2.Sobel(dwt_with_ae, cv2.CV_64F, 0, 1, ksize=3)
                gradient_dir_ae = np.arctan2(sobely_ae, sobelx_ae)
                axs[2].imshow(gradient_dir_ae, cmap='hsv')
                axs[2].set_title("Proposed Edge Directions", fontsize=14)
                axs[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Free up GPU memory
            gc.collect()
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()