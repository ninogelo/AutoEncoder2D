import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import keras
import plaidml.keras
plaidml.keras.install_backend()  # Use PlaidML backend for Keras

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Activation, MaxPooling2D, UpSampling2D  
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

# Autoencoder Model Definition
def create_autoencoder():
    input_img = Input(shape=(256, 256, 3))

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='valid')(x)  # Use 'valid' padding
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='valid')(x)  # Use 'valid' padding

    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

# Custom Data Generator for Autoencoder
def autoencoder_data_generator(generator):
    while True:
        x = next(generator)
        yield (x, x)  # Use the same data as input and target

# Training Function
def train_autoencoder(autoencoder):
    # Configure the ImageDataGenerator for loading images
    data_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Load the images from the specified directory
    data_directory = "./chest_xray_pneumonia/chest_xray/train"
    generator = data_gen.flow_from_directory(
        data_directory,
        target_size=(256, 256),
        batch_size=32,
        class_mode=None  # Indicate that there are no labels
    )

    # Use the custom data generator
    data_generator = autoencoder_data_generator(generator)

    # Use fit_generator to train the model
    history = autoencoder.fit_generator(
        data_generator,
        steps_per_epoch=len(generator),
        epochs=20
    )

    # Save the trained model
    autoencoder.save('autoencoder_model.h5')
    print("Model saved to autoencoder_model.h5")

    # Plot the training loss
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# Main Function
def main():
    autoencoder = create_autoencoder()
    train_autoencoder(autoencoder)

if __name__ == "__main__":
    main()
