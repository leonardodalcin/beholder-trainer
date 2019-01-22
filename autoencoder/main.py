
import matplotlib.pyplot as plt
from tensorflow import keras
from keras_contrib.losses import DSSIMObjective

Input = keras.layers.Input

Dense = keras.layers.Dense
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
UpSampling2D = keras.layers.UpSampling2D
Model = keras.models.Model
K = keras.backend

input_img = Input(shape=(256, 256, 1))  # adapt this if using `channels_first` image data format
x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 256-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss=DSSIMObjective())

import numpy as np
import utils.Image_loader as il
import glob2
import cv2
from sklearn.model_selection import train_test_split
good_path = "/home/leonardo/PycharmProjects/beholder-trainer/utils/empty"
bad_path = "/media/leonardo/Images/classes/well_made/datasets/not_empty"

print("getting good paths")
good_paths = glob2.glob(good_path  + "//*.jpg")
print("getting bad paths")
# bad_paths = glob2.glob(bad_path  + "//*.pgm")

print("loading train set")

images = [(cv2.resize(cv2.imread(file, 0), (256,256)) / 255) for file in good_paths]

x_train, x_test = train_test_split(images, test_size=0.1, random_state = 15)

x_train = np.array(x_train)
x_train = x_train.astype('float32')

x_test = np.array(x_test)
x_test = x_test.astype('float32')

x_train = np.reshape(x_train, (len(x_train), 256, 256, 1))
x_test = np.reshape(x_test, (len(x_test), 256, 256, 1))

print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=30,
                batch_size=16,
                shuffle=True,
                validation_data=(x_test, x_test))

autoencoder.save('autoencoder_mssi_loss.h5')

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(256, 256))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(256, 256))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()