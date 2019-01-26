from autoencoder.Autoencoder import Autoencoder
from skimage.measure import compare_ssim, compare_mse, compare_nrmse, compare_psnr
import utils.Image_loader as il
import numpy as np
from sklearn import svm
from joblib import dump, load

good_path = "/home/leonardo/PycharmProjects/beholder-trainer/imgs/good_empty"
bad_path = "/home/leonardo/PycharmProjects/beholder-trainer/imgs/bad_empty"

def prepare_imgs_array(imgs, data_type, divided_by, reshape_shape):
    imgs = np.array(imgs)
    if(data_type):
        imgs = imgs.astype(data_type)
    if(divided_by):
        imgs = imgs/255
    if(reshape_shape):
        imgs = np.reshape(imgs, reshape_shape)
    return imgs

def get_features_array(imgs_true, imgs_pred):
    return np.array(list(map(lambda x, y: (compare_ssim(x.reshape(256, 256), y.reshape(256, 256)), compare_mse(x.reshape(256, 256), y.reshape(256, 256)),
                           compare_nrmse(x.reshape(256, 256), y.reshape(256, 256)), compare_psnr(x.reshape(256, 256), y.reshape(256, 256))), imgs_true, imgs_pred)))

def get_wrong_prediction_filename(filenames, predictions, true_label):
    wrong_filenames = []
    for index, prediction in enumerate(predictions):
        if(prediction != true_label):
            wrong_filenames.append(filenames[index])

    return wrong_filenames
autoencoder = Autoencoder()

bad_paths, bad_images = il.get_images_in_folder(bad_path, "jpg")
bad_images = prepare_imgs_array(bad_images, "float32", 255, (len(bad_images), 256, 256, 1))
bad_images_autoencoded = autoencoder.predict(bad_images)
bad_features = get_features_array(bad_images,bad_images_autoencoded)
np.save("bad_features", bad_features)

clf = load('/home/leonardo/PycharmProjects/beholder-trainer/classifiers/svm3.joblib')
# predictions = clf.predict(bad_features)
# print("bad predictions")
# print(predictions)

del bad_features
del bad_images_autoencoded
del bad_images

good_paths, good_images = il.get_images_in_folder(good_path, "jpg")
good_images = prepare_imgs_array(good_images, "float32", 255, (len(good_images), 256, 256, 1))
good_images_autoencoded = autoencoder.predict(good_images)
good_features = get_features_array(good_images,good_images_autoencoded)
np.save("good_features", good_features)
predictions = clf.predict(good_features)
print("good predictions")
print(predictions)

wrong_filenames = get_wrong_prediction_filename(good_paths, predictions, 1.)
print(len(wrong_filenames))
from Image import Image

wrong_predicted_images = il.get_images(wrong_filenames)


for image in wrong_predicted_images:
    im = Image(image)
    im.show()
