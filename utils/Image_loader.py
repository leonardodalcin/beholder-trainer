import cv2
import os
import glob
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean

def getImagePaths(folder, imgExts):
    imagePaths = []
    for x in os.listdir(folder):
        xPath = os.path.join(folder, x)
        if os.path.splitext(xPath)[1] in imgExts:
            imagePaths.append(xPath)
    return imagePaths

def open_image_sklearn(path, size):
    img = imread(path, as_gray=True)
    if(size):
        return resize(img, size, anti_aliasing=True)
    else:
        return img

def open_image_opencv(path):
    return cv2.imread(path, 0)
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