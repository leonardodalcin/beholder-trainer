from models.cable_tie_length.Cable_tie_length import Cable_tie_length
import numpy as np
import cv2
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

models_dictionary = {
	# "empty_mould": Empty_mould(),
	"cable_tie_length": Cable_tie_length()
}


class Classifier:
	model = None

	def __init__(self, model_type):
		try:
			models_dictionary[model_type]
		except NameError:
			print("No model with name " + model_type)
		else:
			self.model = models_dictionary[model_type]

	# train the network
	def train(self, train_directory, validation_directory):
		batch_size = 4

		# this is the augmentation configuration we will use for training
		train_datagen = ImageDataGenerator(
			rescale=1. / 255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True)

		# this is the augmentation configuration we will use for testing:
		# only rescaling
		test_datagen = ImageDataGenerator(rescale=1. / 255)

		# this is a generator that will read pictures found in
		# subfolers of 'data/train', and indefinitely generate
		# batches of augmented image data
		train_generator = train_datagen.flow_from_directory(
			train_directory,  # this is the target directory
			target_size=(150, 150),  # all images will be resized to 150x150
			batch_size=batch_size,
			class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

		# this is a similar generator, for validation data
		validation_generator = test_datagen.flow_from_directory(
			validation_directory,
			target_size=(150, 150),
			batch_size=batch_size,
			class_mode='binary')

		self.model.fit_generator(
			train_generator,
			steps_per_epoch=200 // batch_size,
			epochs=10,
			validation_data=validation_generator,
			validation_steps=100 // batch_size)

		model.save_weights('10_epochs.h5')
		return

	# evaluates an array of images and returns an array of predictions
	def predict(self, images_folder_path):
		images_to_be_predicted = np.array([cv2.resize(cv2.imread(file)) for file in glob.glob(
			images_folder_path + "/*.pgm")])[0:32]
		predictions = self.model.predict(images_to_be_predicted)
		return predictions

