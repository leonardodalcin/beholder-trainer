
import matplotlib.pyplot as plt
from tensorflow import keras
from skimage.measure import compare_ssim, compare_mse, compare_nrmse, compare_psnr
from keras_contrib.losses import DSSIMObjective
Input = keras.layers.Input

Dense = keras.layers.Dense
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
UpSampling2D = keras.layers.UpSampling2D
Model = keras.models.Model()
K = keras.backend
import numpy as np
import utils.Image_loader as il
import glob2
import cv2
from sklearn.model_selection import train_test_split
good_path = "/home/leonardo/PycharmProjects/beholder-trainer/utils/empty"
bad_path = "/media/leonardo/Images/classes/well_made/datasets/not_empty"

smiloss = DSSIMObjective()
def rmse_loss(y_true,y_pred):
    return smiloss(y_true,y_pred) + K.sqrt(K.mean(K.square(y_pred - y_true)))

def rmse_smi_loss(y_true,y_pred):
    return smiloss(y_true,y_pred) + K.sqrt(K.mean(K.square(y_pred - y_true)))

print("getting good paths")
good_paths = glob2.glob(good_path  + "//*.jpg")[:11]
print("getting bad paths")
bad_paths = glob2.glob(bad_path  + "//*.pgm")[:11]

print("loading train set")
# [array([[[[ 0.0045734 ],
#          [-0.00805628],
#          [ 0.07753198],
#          [ 0.13515475],
#          [ 0.06268791],

print("BAD: ")
images = []
for path in bad_paths:
    try:
        im = cv2.resize(cv2.imread(path, 0), (256, 256))
        images.append(im)
    except:
        print("e")

x_test = np.array(images)
x_test = x_test.astype('float32') / 255
x_test = np.reshape(x_test, (len(x_test), 256, 256, 1))
print(x_test.shape)

autoencoder = keras.models.load_model("./autoencoder_conv_rmse_mssi_loss.h5", custom_objects={'rmse_smi_loss': rmse_smi_loss})
autoencoder.summary()
decoded_imgs = autoencoder.predict(x_test, verbose=1)
compare_function_names = ["compare_ssim", "compare_mse", "compare_nrmse", "compare_psnr"]
compare_functions = [compare_ssim, compare_mse, compare_nrmse, compare_psnr]
for image_index, decoded_img in enumerate(decoded_imgs):
    print("Image: " + str(image_index))
    for index, function in enumerate(compare_functions):
        print(compare_function_names[index] + ": " + str(function(x_test[image_index].reshape(256, 256), decoded_img.reshape(256, 256))))

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(256, 256))
    plt.gray()
    ax.set_title("compare_ssim: " + str(compare_ssim(x_test[i].reshape(256, 256), decoded_imgs[i].reshape(256, 256)))
                 + "\ncompare_mse: " + str(compare_mse(x_test[i].reshape(256, 256), decoded_imgs[i].reshape(256, 256)))
                 + "\ncompare_nrmse: " + str(compare_nrmse(x_test[i].reshape(256, 256), decoded_imgs[i].reshape(256, 256)))
                 + "\ncompare_psnr: " + str(compare_psnr(x_test[i].reshape(256, 256), decoded_imgs[i].reshape(256, 256))), fontsize=5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(256, 256))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

print("GOOD: ")
images = []
for path in good_paths:
    try:
        im = cv2.resize(cv2.imread(path, 0), (256, 256))
        images.append(im)
    except:
        print("e")

x_test = np.array(images)
x_test = x_test.astype('float32') / 255
x_test = np.reshape(x_test, (len(x_test), 256, 256, 1))
print(x_test.shape)

autoencoder = keras.models.load_model("./autoencoder_conv_rmse_mssi_loss.h5",custom_objects={'rmse_smi_loss': rmse_smi_loss})
decoded_imgs = autoencoder.predict(x_test, verbose=1)
compare_function_names = ["compare_ssim", "compare_mse", "compare_nrmse", "compare_psnr"]
compare_functions = [compare_ssim, compare_mse, compare_nrmse, compare_psnr]
for image_index, decoded_img in enumerate(decoded_imgs):
    print("Image: " + str(image_index))
    for index, function in enumerate(compare_functions):
        print(compare_function_names[index] + ": " + str(function(x_test[image_index].reshape(256, 256), decoded_img.reshape(256, 256))))
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(256, 256))
    plt.gray()
    ax.set_title("compare_ssim: " + str(compare_ssim(x_test[i].reshape(256, 256), decoded_imgs[i].reshape(256, 256)))
                 + "\ncompare_mse: " + str(compare_mse(x_test[i].reshape(256, 256), decoded_imgs[i].reshape(256, 256)))
                 + "\ncompare_nrmse: " + str(compare_nrmse(x_test[i].reshape(256, 256), decoded_imgs[i].reshape(256, 256)))
                 + "\ncompare_psnr: " + str(compare_psnr(x_test[i].reshape(256, 256), decoded_imgs[i].reshape(256, 256))),  fontsize=5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(256, 256))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


print("Single: ")
path = "/home/leonardo/Pictures/welcome.jpg"
img = [cv2.resize(cv2.imread(path, 0), (256, 256))]

x_test = np.array(img)
x_test = x_test.astype('float32') / 255
x_test = np.reshape(x_test, (len(x_test), 256, 256, 1))
print(x_test.shape)

autoencoder = keras.models.load_model("./autoencoder_conv_rmse_mssi_loss.h5",custom_objects={'rmse_smi_loss': rmse_smi_loss})
decoded_imgs = autoencoder.predict(x_test, verbose=1)

plt.figure(figsize=(20, 4))
# display original
ax = plt.subplot(2, 1, 1)
plt.imshow(x_test[0].reshape(256, 256))
plt.gray()
ax.set_title("compare_ssim: " + str(compare_ssim(x_test[0].reshape(256, 256), decoded_imgs[0].reshape(256, 256)))
             + "\ncompare_mse: " + str(compare_mse(x_test[0].reshape(256, 256), decoded_imgs[0].reshape(256, 256)))
             + "\ncompare_nrmse: " + str(compare_nrmse(x_test[0].reshape(256, 256), decoded_imgs[0].reshape(256, 256)))
             + "\ncompare_psnr: " + str(compare_psnr(x_test[0].reshape(256, 256), decoded_imgs[0].reshape(256, 256))),  fontsize=5)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# display reconstruction
ax = plt.subplot(2, 1, 2)
plt.imshow(decoded_imgs[0].reshape(256, 256))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

