from autoencoder.loss_functions import loss_functions as lf
from tensorflow import keras

class Autoencoder:

    def __init__(self, model_path = "/home/leonardo/PycharmProjects/beholder-trainer/autoencoder/models/autoencoder_conv_rmse_mssi_loss.h5"):
        self.model = keras.models.load_model(model_path, custom_objects={'rmse_smi_loss': lf.rmse_smi_loss})


    def predict(self, to_predict):
        return self.model.predict(to_predict, verbose=1)


