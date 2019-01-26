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


autoencoder = Autoencoder()

bad_paths, bad_images = il.get_images_in_folder(bad_path, "jpg")
bad_images = prepare_imgs_array(bad_images, "float32", 255, (len(bad_images), 256, 256, 1))
bad_images_autoencoded = autoencoder.predict(bad_images)
bad_features = get_features_array(bad_images,bad_images_autoencoded)
print("Bad features: ", bad_features)

good_paths, bad_images = il.get_images_in_folder(good_path, "jpg")
bad_images = prepare_imgs_array(bad_images, "float32", 255, (len(bad_images), 256, 256, 1))
bad_images_autoencoded = autoencoder.predict(bad_images)
good_features = get_features_array(bad_images,bad_images_autoencoded)
print("Good features: ", good_features)

y = np.concatenate([np.ones(len(good_features)),np.zeros(len(bad_features))])
x = np.concatenate([good_features, bad_features])

clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.1, verbose=False)
clf.fit(x, y)
dump(clf, 'svm3.joblib')



exit(0)

