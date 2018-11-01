import pathlib
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from os import walk
from glob import glob
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

checkpoint_path = "third_try.h5"
class_names = ["good", "bad"]



import glob

test_images = np.array([cv2.resize(cv2.imread(file),(150,150)) for file in glob.glob("/Users/leonardodalcin/PycharmProjects/image_classifier/data/train/bad/*.png")])

test_labels = np.zeros((len(test_images),), dtype=int)

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

predictions = model.predict(test_images)

for prediction in predictions:
    print(prediction[0])