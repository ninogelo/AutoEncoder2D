import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load and preprocess a custom image
def load_image(filepath):
    img = io.imread(filepath, as_gray=True)  # Load the image in grayscale
    img = img_as_float(img)  # Convert to float
    img = img[:256, :256]  # Resize to 256x256 if necessary
    return img

# Define Haar wavelet filters
h = np.array([1, 1]) / np.sqrt(2)  # Low-pass filter
g = np.array([1, -1]) / np.sqrt(2)  # High-pass filter

# Function to apply 1D filtering along rows or columns
def apply_filter(img, f, axis=0):
    return np.apply_along_axis(lambda row: np.convolve(row, f, mode='same'), axis, img)

# Main processing function
def process_image(filepath):
    # Load the custom image
    X = load_image(filepath)

    # 2D Haar wavelet decomposition (manual approximation)
    L = apply_filter(apply_filter(X, h, axis=0), h, axis=1)  # Approximation
    H = apply_filter(apply_filter(X, h, axis=0), g, axis=1)  # Horizontal detail
    V = apply_filter(apply_filter(X, g, axis=0), h, axis=1)  # Vertical detail
    D = apply_filter(apply_filter(X, g, axis=0), g, axis=1)  # Diagonal detail

    # Simulate phase shifts by swapping horizontal and vertical detail coefficients
    H_shifted = V  # Assign vertical details to horizontal
    V_shifted = H  # Assign horizontal details to vertical

    # Reconstruct the image with phase-shifted coefficients
    X_phase_shifted = apply_filter(apply_filter(L, h, axis=0), h, axis=1) + \
                      apply_filter(apply_filter(H_shifted, h, axis=0), g, axis=1) + \
                      apply_filter(apply_filter(V_shifted, g, axis=0), h, axis=1) + \
                      apply_filter(apply_filter(D, g, axis=0), g, axis=1)

    # Calculate Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)
    mse_val = mean_squared_error(X, X_phase_shifted)
    psnr_val = peak_signal_noise_ratio(X, X_phase_shifted)

    # Calculate the difference in coefficients between the original and phase-shifted images
    coeff_diff = np.abs(H - H_shifted) + np.abs(V - V_shifted)

    # Flatten the coefficient difference array for tabular representation
    coeff_diff_flat = coeff_diff.flatten()

    # Select top 10 highest differences for the table
    top_diff_indices = np.argsort(-coeff_diff_flat)[:10]  # Sort in descending order
    top_diff_values = coeff_diff_flat[top_diff_indices]

    # Create a table using pandas DataFrame
    table_data = pd.DataFrame({
        'Index': top_diff_indices,
        'Difference in Coefficients': top_diff_values
    })

    # Display metrics
    print(f"Mean Squared Error (MSE): {mse_val:.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_val:.4f} dB")

    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Original image
    axs[0, 0].imshow(X, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Phase-shifted image
    axs[0, 1].imshow(X_phase_shifted, cmap='gray')
    axs[0, 1].set_title('Phase-Shifted Image')
    axs[0, 1].axis('off')

    # Difference in wavelet coefficients
    im = axs[0, 2].imshow(coeff_diff, cmap='hot', interpolation='nearest')
    axs[0, 2].set_title('Difference in Wavelet Coefficients')
    fig.colorbar(im, ax=axs[0, 2], orientation='vertical')

    # Histogram of the original image
    axs[1, 0].hist(X.ravel(), bins=256, color='black')
    axs[1, 0].set_title('Histogram of Original Image')

    # Histogram of the phase-shifted image
    axs[1, 1].hist(X_phase_shifted.ravel(), bins=256, color='black')
    axs[1, 1].set_title('Histogram of Phase-Shifted Image')

    # Table of top differences
    axs[1, 2].axis('off')  # Turn off the axis
    table = axs[1, 2].table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
    table.scale(1.5, 1.5)  # Adjust the scale for better readability
    axs[1, 2].set_title('Top Differences in Wavelet Coefficients')

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Function to open file dialog and upload an image
def upload_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if filepath:
        process_image(filepath)

# Create Tkinter GUI
root = tk.Tk()
root.title("Image Upload for Wavelet Decomposition")

# Upload button
upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=20)

# Run the application
root.mainloop()
