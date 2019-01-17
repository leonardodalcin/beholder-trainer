
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('/media/leonardo/Images/full_mould_good/dataset_01/classes/good/1e8de054-7a32-4190-baa3-5663e6f0b268good.jpg',0)
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
for k in kp:
    print(k)
plt.imshow(img), plt.show()