from datetime import datetime
from matplotlib import pyplot as plt
import cv2
import os

class Image():
    image = None

    def __init__(self, image=None, path=None):
        if (path):
            self.image = cv2.imread(path, 0)
        else:
            self.image = image

    def show(self):
        plt.imshow(self.image, cmap="gray", interpolation='bicubic')
        plt.show()

    def save(self, label = ""):
        now = datetime.now()
        dirName = now.strftime("%d-%m-%Y") + str(label)
        fileName = now.strftime("%X")
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        cv2.imwrite(dirName + "/" + fileName + ".jpg", self.image)

    def rotate(self, degrees):
        (height, width) = self.image.shape[:2]
        center = (height / 2, width / 2)
        rotationMatrix = cv2.getRotationMatrix2D(center, degrees, scale=1)
        self.image = cv2.warpAffine(self.image, rotationMatrix, (width, height))
