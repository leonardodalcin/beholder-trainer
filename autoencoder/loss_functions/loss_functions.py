from tensorflow import keras
from keras import backend as K
from keras_contrib.losses import DSSIMObjective
from keras.losses import mean_squared_error

smi_loss = DSSIMObjective()

def rmse_loss(y_true,y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def rmse_smi_loss(y_true,y_pred):
    return smi_loss(y_true, y_pred) + rmse_loss(y_true, y_pred)

