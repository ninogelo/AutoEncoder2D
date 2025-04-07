import os
import kaggle

# Specify the Kaggle dataset name
dataset_name = "paultimothymooney/chest-xray-pneumonia"

# Set the output path for the dataset
output_path = "C:/Users/Gelo/PycharmProjects/AutoEncoder/chest_xray_pneumonia"
os.makedirs(output_path, exist_ok=True)  # Create the directory if it doesn't exist

# Use the Kaggle API to download and unzip the dataset
kaggle.api.dataset_download_files(dataset_name, path=output_path, unzip=True)

print("Dataset downloaded and unzipped to:", output_path)
