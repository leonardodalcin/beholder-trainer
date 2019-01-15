import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
from skimage.io import imread

def get_hog_vector(img):
    fd, hog_image = hog(img, block_norm="L2-Hys", orientations=16, pixels_per_cell=(8, 8),
                        cells_per_block=(3, 3), visualize=True, multichannel=False)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    #
    # ax1.axis('off')
    # ax1.imshow(img, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    #
    # # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    #
    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()
    return fd