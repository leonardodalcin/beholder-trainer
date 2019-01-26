import cv2
import os
import glob
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
import glob2
import numpy as np
import uuid

from joblib import Parallel, delayed


good_path = "/home/leonardo/PycharmProjects/beholder-trainer/utils/empty"
bad_path = "/media/leonardo/Images/classes/well_made/datasets/not_empty"

def get_sample():
    return open_image_sklearn("/media/leonardo/Images/full_mould_good/dataset_01/classes/good/1e8de054-7a32-4190-baa3-5663e6f0b268good.jpg")

def getImagePaths(folder, imgExts):
    imagePaths = []
    for x in os.listdir(folder):
        xPath = os.path.join(folder, x)
        if os.path.splitext(xPath)[1] in imgExts:
            imagePaths.append(xPath)
    return imagePaths

def open_image_sklearn(path, size = False):
    img = imread(path, as_gray=True)
    if(size):
        return resize(img, size, anti_aliasing=True)
    else:
        return img

def inter_open_image_opencv(tup):
    return open_image_opencv(*tup)

def open_image_opencv(path, resize_size=None):
    img = cv2.imread(path, 0)
    try:
        if img is not None and img.size is not 0:
            if(resize_size):
                return cv2.resize(img, resize_size)
            else:
                return img
        else:
            return 0
    except:
        return 0
# read images in a folder
# return list of images and labels
def getDataset(folder, classLabel):
    images = []
    labels = []
    imagePaths = getImagePaths(folder, ['.jpg', '.png', '.jpeg'])
    for imagePath in imagePaths:
        print(imagePath)
        im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        images.append(im)
        labels.append(classLabel)
    return images, labels

def get_images_in_folder(folder,extension,resize_size=None, img_class = None):
    paths = glob2.glob(folder + "//*." + extension)
    print("Loading " + str(len(paths)) + " imgs")
    imgs = []
    for path in paths:
        img = open_image_opencv(path, resize_size)
        if img is not 0:
            imgs.append(img)
            if(img_class):
                cv2.imwrite("/home/leonardo/PycharmProjects/beholder-trainer/imgs/" + img_class + "/" + str(uuid.uuid4()) + ".jpg", img)
    return (paths, np.array(imgs))

def get_images(paths,resize_size=None, img_class = None):
    print("Loading " + str(len(paths)) + " imgs")
    imgs = []
    for path in paths:
        img = open_image_opencv(path, resize_size)
        if img is not 0:
            imgs.append(img)
            if(img_class):
                cv2.imwrite("/home/leonardo/PycharmProjects/beholder-trainer/imgs/" + img_class + "/" + str(uuid.uuid4()) + ".jpg", img)
    return np.array(imgs)

