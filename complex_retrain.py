import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import keras
import plaidml.keras
plaidml.keras.install_backend()  # Use PlaidML as the Keras backend

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU
from keras import regularizers, initializers
from keras.optimizers import RMSprop
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import pywt
from keras.utils import Sequence
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

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
    def __init__(self, generators, batch_size):
        self.generators = generators
        self.batch_size = batch_size

    def __len__(self):
        return sum([len(gen) for gen in self.generators])

    def __getitem__(self, index):
        generator_index = 0
        while index >= len(self.generators[generator_index]):
            index -= len(self.generators[generator_index])
            generator_index += 1
        batch = self.generators[generator_index][index]
        dwt_batch = preprocess_images_with_dwt(batch)
        return dwt_batch, dwt_batch  # Return (input, target)

# Autoencoder Model Definition
def create_autoencoder():
    input_img = Input(shape=(128, 128, 3))  # Updated input shape for LL coefficients (128x128)

    # Encoder with non-separable 2D filter bank
    x = Conv2D(32, (3, 3), padding='same',
               kernel_initializer=initializers.constant(create_non_separable_filters((3, 3, 3, 32))),
               kernel_regularizer=regularizers.l2(0.001))(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='valid')(x)
    
    x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='valid')(x)

    # Decoder
    x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # The autoencoder model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=RMSprop(learning_rate=1e-4), loss='mean_squared_error')
    return autoencoder

# Modified Training Function with Debug Prints
def train_autoencoder(autoencoder, model_path='autoencoder_model_rgb_with_cayley_dwt.h5'):
    # Load the previous model if it exists
    if os.path.exists(model_path):
        autoencoder = load_model(model_path)
        print(f"Loaded existing model from {model_path}")
    else:
        print("No existing model found, starting training from scratch.")

    # Configure the ImageDataGenerator for loading images with augmentation
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
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
    
    # Check data generator length
    num_batches = len(data_generator)
    print(f"Data generator contains {num_batches} batches")

    # Callbacks for training
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

    # Train the model using fit
    if num_batches > 0:
        print("Starting training...")
        history = autoencoder.fit(
            data_generator,
            steps_per_epoch=num_batches,
            epochs=10,  # Reduced for debugging
            callbacks=[lr_scheduler, early_stopping]
        )

        # Save the trained model
        autoencoder.save(model_path)
        print(f"Model saved to {model_path}")

        # Plot the training loss
        plt.plot(history.history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    else:
        print("No data available for training")


# Main Function
def main():
    # Load existing model if it exists, otherwise create a new one
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
