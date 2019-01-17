import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
import utils.Image_loader as il
import segmentation.threshold as t
img = il.get_sample()
img = t.otsu(img)

# Find contours at a constant value of 0.8
perimeter = measure.perimeter(img, 8)
print(perimeter)