input_img = Input(shape=(256, 256,1))  # adapt this if using `channels_first` image data format
x = keras.layers.Flatten()(input_img)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(256*256, activation='sigmoid')(x)
# =================================================================
# input_1 (InputLayer)         (None, 256, 256)          0
# _________________________________________________________________
#   File "/home/leonardo/Downloads/pycharm-2018.3.3/helpers/pydev/_pydev_bundle/pydev_umd.py", line 197, in runfile
# flatten (Flatten)            (None, 65536)             0
# _________________________________________________________________
# dense (Dense)                (None, 128)               8388736
# _________________________________________________________________
# dense_1 (Dense)              (None, 64)                8256
# _________________________________________________________________
# dense_2 (Dense)              (None, 32)                2080
# _________________________________________________________________
# dense_3 (Dense)              (None, 64)                2112
# _________________________________________________________________
# dense_4 (Dense)              (None, 128)               8320
# _________________________________________________________________
# dense_5 (Dense)              (None, 65536)             8454144
# _________________________________________________________________
# reshape (Reshape)            (None, 256, 256)          0
# =================================================================
x = keras.layers.Reshape((256,256,1))(x)