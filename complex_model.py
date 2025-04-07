import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import keras
import plaidml.keras
plaidml.keras.install_backend()  # Use PlaidML as the Keras backend

from keras.models import Model
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras import regularizers, initializers
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import pywt
from keras.utils import Sequence

# Helper function for Cayley transform (for enforcing orthogonality)
def cayley_transform(M):
    """
    Apply the Cayley transform to the matrix to enforce orthogonality.
    This transforms a matrix M into an orthogonal matrix.
    """
    I = np.identity(M.shape[0])
    return np.linalg.inv(I + M) @ (I - M)

# Function to create non-separable filters with symmetry and vanishing moments
def create_non_separable_filters(shape):
    """
    Create non-separable filters with orthogonality enforced by Cayley transform.
    This function will iterate over each 2D filter and apply the transform individually.
    """
    filters = np.random.randn(*shape)  # Initialize filters randomly

    # Apply Cayley transform to enforce orthogonality on each 2D filter slice
    for i in range(shape[2]):
        for j in range(shape[3]):
            filters[:, :, i, j] = cayley_transform(filters[:, :, i, j])

    return filters

# Function to preprocess images with DWT
def preprocess_images_with_dwt(image_batch):
    batch_size, height, width, channels = image_batch.shape
    dwt_batch = np.zeros((batch_size, height // 2, width // 2, channels))  # Adjust shape for LL coefficients

    for i in range(batch_size):
        for c in range(channels):
            channel = image_batch[i, :, :, c]
            LL, (LH, HL, HH) = pywt.dwt2(channel, 'haar')
            dwt_batch[i, :, :, c] = LL

    return dwt_batch

# Custom data generator class
class DWTDataGenerator(Sequence):
    def __init__(self, generator, batch_size):
        self.generator = generator
        self.batch_size = batch_size

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        batch = self.generator[index]
        dwt_batch = preprocess_images_with_dwt(batch)
        return dwt_batch, dwt_batch  # Return (input, target)

# Autoencoder Model Definition
def create_autoencoder():
    input_img = Input(shape=(128, 128, 3))  # Updated input shape for LL coefficients (128x128)

    # Encoder with non-separable 2D filter bank
    x = Conv2D(16, (3, 3), padding='same', activation='relu',
               kernel_initializer=initializers.constant(create_non_separable_filters((3, 3, 3, 16))),
               kernel_regularizer=regularizers.l2(0.01))(input_img)
    x = MaxPooling2D((2, 2), padding='valid')(x)
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='valid')(x)

    # Decoder
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # The autoencoder model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

# Training Function
def train_autoencoder(autoencoder, model_path='autoencoder_model_rgb_with_cayley_dwt.h5'):
    if os.path.exists(model_path):
        autoencoder = load_model(model_path)
        print(f"Loaded existing model from {model_path}")
    else:
        print("No existing model found, starting training from scratch.")
    # Configure the ImageDataGenerator for loading images
    data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    data_directory = "./Medical-imaging-dataset"
    
    # Load the images from the specified directory
    generator = data_gen.flow_from_directory(
        data_directory,
        target_size=(256, 256),
        batch_size=32,
        class_mode=None
    )

    # Create an instance of the custom data generator
    data_generator = DWTDataGenerator(generator, batch_size=32)

    # Train the model using fit_generator
    history = autoencoder.fit_generator(
        data_generator,
        steps_per_epoch=len(data_generator),
        epochs=150
    )

    # Save the trained model
    autoencoder.save('autoencoder_model_rgb_with_cayley_dwt.h5')
    print("Model saved to autoencoder_model_rgb_with_cayley_dwt.h5")

    # Plot the training loss
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# Main Function
def main():
    model_path = 'autoencoder_model_rgb_with_cayley_dwt.h5'
    if os.path.exists(model_path):
        autoencoder = load_model(model_path)
        print("Loaded existing model.")
    else:
        autoencoder = create_autoencoder()
        print("Created a new model.")

    # Train the model
    train_autoencoder(autoencoder, model_path)

if __name__ == "__main__":
    main()
