import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage

from Image import Image
# Generate noisy synthetic data
data = skimage.img_as_float(binary_blobs(length=128, seed=1))
sigma = 0.35
data += np.random.normal(loc=0, scale=sigma, size=data.shape)
from skimage.feature import shape_index
from skimage import data
import matplotlib.pyplot as plt
import utils.Image_loader as il

path = "/media/leonardo/Images/full_mould_good/dataset_01/classes/good"
bad_path = "/media/leonardo/Images/full_mould_bad/classes/bad"

good_path = bad_path

good_paths = il.getImagePaths(bad_path, ".jpg")

pos_imgs = []
for path in good_paths:
    pos_imgs.append(il.open_image_sklearn(path))
from skimage.segmentation import felzenszwalb
from skimage.data import coffee
img = pos_imgs[0]
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2gray

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

from skimage.segmentation import slic
from skimage.data import astronaut
segments = slic(img, n_segments=5, compactness=10.0, max_iter=10, sigma=0, spacing=None, multichannel=False, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False)

# Image(segments).show()
ax[0, 1].imshow(mark_boundaries(img, segments))
plt.tight_layout()
plt.show()

