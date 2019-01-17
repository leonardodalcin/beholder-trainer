import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
import utils.Image_loader as il
import segmentation.threshold as t
import measure.find_countours as fc

img = il.get_sample()
countour = fc.get_contour(img)

polygon = measure.approximate_polygon(countour, 0.8)
print(polygon)

import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import ellipse
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon


hand = countour

# subdivide polygon using 2nd degree B-Splines
new_hand = hand.copy()
for _ in range(5):
    new_hand = subdivide_polygon(new_hand, degree=2, preserve_ends=True)

# approximate subdivided polygon with Douglas-Peucker algorithm
appr_hand = approximate_polygon(new_hand, tolerance=0.02)

print("Number of coordinates:", len(hand), len(new_hand), len(appr_hand))

fig, (ax1) = plt.subplots()

ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Input image')
# ax1.plot(hand[:, 1], hand[:, 0])
ax1.plot(new_hand[:, 1], new_hand[:, 0])
# ax1.plot(appr_hand[:, 1], appr_hand[:, 0])

plt.show()