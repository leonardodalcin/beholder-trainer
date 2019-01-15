import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import glob
import utils.Image_loader as image_loader
import featureextractors.HOG as hog
from sklearn.ensemble import IsolationForest

good_path = "/media/leonardo/Images/full_mould_good/dataset_01/classes/good"
bad_path = "/media/leonardo/Images/full_mould_bad/classes/bad"

good_paths = image_loader.getImagePaths(good_path, ".jpg")
bad_paths = image_loader.getImagePaths(bad_path, ".jpg")

pos_imgs = []
for path in good_paths:
    pos_imgs.append(image_loader.open_image_sklearn(path, (300, 100)))

neg_imgs = []
for path in bad_paths:
    neg_imgs.append(image_loader.open_image_sklearn(path, (300, 100)))

pos_hogs = []
for img in pos_imgs:
    pos_hogs.append(hog.get_hog_vector(img))

neg_hogs = []
for img in neg_imgs:
    neg_hogs.append(hog.get_hog_vector(img))

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
X_train = pos_hogs
X_test = pos_hogs
X_outliers = neg_hogs

clf = IsolationForest(n_estimators=4, behaviour='new',contamination=0, random_state=42)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
# n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

print(y_pred_train)

