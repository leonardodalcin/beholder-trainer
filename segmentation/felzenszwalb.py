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
from skimage.segmentation import mark_boundaries

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
segments = felzenszwalb(img, scale=400, sigma=0.95, min_size=100)
Image(mark_boundaries(img, segments)).show()
print("Felzenszwalb number of segments: {}".format(len(np.unique(segments))))

