# MUST BE FIRST - Configure backend before any Keras imports
import plaidml.keras
plaidml.keras.install_backend()

import numpy as np
import pywt
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Add
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class CayleyWavelet(pywt.Wavelet):
    def __init__(self):
        h = [0.48296, 0.83652, 0.22414, -0.12941]
        g = [-0.12941, -0.22414, 0.83652, -0.48296]
        pywt.Wavelet.__init__(self, name='CayleyWavelet', filter_bank=(h, g, h[::-1], g[::-1]))

def build_autoencoder():
    input_img = Input(shape=(64, 64, 1))
    
    # Encoder
    x1 = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(2)(x1)
    x2 = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(2)(x2)
    
    # Decoder
    x = Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
    x = UpSampling2D(2)(x)
    x = Add()([x, x2])
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    x = Add()([x, x1])
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder():
    autoencoder = build_autoencoder()
    datagen = ImageDataGenerator(rescale=1./255)
    
    def ll_generator(generator):
        for img in generator:
            gray = 0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]
            coeffs = pywt.wavedec2(gray, CayleyWavelet(), level=2)
            ll = coeffs[0][..., np.newaxis]
            yield (ll, ll)
    
    train_gen = datagen.flow_from_directory(
        './Medical-Images',
        target_size=(256, 256),
        batch_size=16,
        class_mode=None,
        color_mode='rgb'
    )
    
    history = autoencoder.fit(
        ll_generator(train_gen),
        steps_per_epoch=len(train_gen),
        epochs=10  # Start with fewer epochs for testing
    )
    
    autoencoder.save('phase_aware_autoencoder.h5')
    plt.plot(history.history['loss'])
    plt.savefig('training_history.png')

if __name__ == "__main__":
    train_autoencoder()