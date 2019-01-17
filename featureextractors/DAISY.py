from skimage.feature import daisy
from skimage import data
import matplotlib.pyplot as plt
import utils.Image_loader as il

path = "/media/leonardo/Images/full_mould_good/dataset_01/classes/good"
bad_path = "/media/leonardo/Images/full_mould_bad/classes/bad"

good_path = bad_path

good_paths = il.getImagePaths(good_path, ".jpg")

pos_imgs = []
for path in good_paths:
    pos_imgs.append(il.open_image_sklearn(path))

for img in pos_imgs:
    descs, descs_img = daisy(img, step=50, radius=150, rings=1, histograms=4,
                             orientations=16, visualize=True)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(descs_img)
    descs_num = descs.shape[0] * descs.shape[1]
    ax.set_title('%i DAISY descriptors extracted:' % descs_num)
    plt.show()
