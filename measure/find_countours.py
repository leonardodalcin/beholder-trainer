import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
import utils.Image_loader as il

r = il.get_sample()

def get_contour(img):
    return measure.find_contours(img, 90, fully_connected="high", positive_orientation="low")[0]
# Find contours at a constant value of 0.8
contours = measure.find_contours(r, 90, fully_connected="high", positive_orientation="low")
contours = [contours[0]]

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()