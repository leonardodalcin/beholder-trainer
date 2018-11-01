from models.Cable_tie_length import Cable_tie_length
from models.Empty_Mould import Empty_mould

models_dictionary = {
	"empty_mould": Empty_mould(),
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
	def train(self):
		return
	# evaluates an array of images and returns an array of predictions
	def predict(self, images):
		return

