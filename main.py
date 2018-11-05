from Classifier import Classifier
bad8_path = "/Users/leonardodalcin/Downloads/bad8"
bad_path4 = "/Users/leonardodalcin/Downloads/bad4-1"
bad_path4_2 = "/Users/leonardodalcin/Downloads/bad4-2"

bad_path0_2 = "/Users/leonardodalcin/Downloads/bad0-2"
bad_path = "/Users/leonardodalcin/Downloads/bad0-1"
train_path = "./data/train"
validation_path = "./data/validation"

cable_tie_length_classifier = Classifier("cable_tie_length")

cable_tie_length_classifier.train(train_path, validation_path)
# predictions = cable_tie_length_classifier.predict(good_path)
#
# for prediction in predictions:
# 	print(prediction)