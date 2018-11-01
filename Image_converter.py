import json
import cv2
import glob
import os

root_path = "/Users/leonardodalcin/Downloads/data"

# finds for image class on filename and return its class name
def get_image_class_from_filename(image_name):
	if image_name.find("empty") > -1:
		return "empty"
	elif image_name.find("not_empty") > -1:
		return "not_empty"
	if image_name.find("good") > -1:
		return "good"
	elif image_name.find("bad") > -1:
		return "bad"
	else:
		return "not_classified"

# creates directories
def create_classes_directories():
	if(not os.path.exists("./good")):
		os.makedirs("./good")
	if (not os.path.exists("./bad")):
		os.makedirs("./bad")
	if (not os.path.exists("./empty")):
		os.makedirs("./empty")
	if (not os.path.exists("./not_empty")):
		os.makedirs("./not_empty")
	if (not os.path.exists("./not_classified")):
		os.makedirs("./not_classified")

# This will delete all files recursively with the given "file_type" starting from the "path" directory.
def delete_by_file_type(path, file_type):
	files = glob.glob(path + "/**/*" + file_type, recursive=True)
	print("Found " + str(len(files)) + " files to be deleted")
	for index, image_name in enumerate(files):
		print("Deleting " + image_name + " press enter to continue")
		pressed_key = input()
		if(pressed_key == ''):
			print("Image " + image_name + " was deleted successfully")
			os.remove(image_name)
		else:
			print("Image " + image_name + " deletion os skipped")


# Do the default
def main():
	# delete_by_file_type(root_path, ".jpg")
	create_classes_directories()
	# Converts all files from given type to another one TODO
	files = glob.glob(root_path + "/**/*.pgm", recursive=True)
	print(str(len(files)) + " to be converted")
	for index, image_name in enumerate(glob.glob(root_path + "/**/*.pgm", recursive=True)):
		print(str(index) + " out of " + str(len(files)))
		print("Filename: " + os.path.basename(image_name))
		image = cv2.imread(image_name, 0)
		new_image_path = "./" + get_image_class_from_filename(image_name) + "/" + os.path.basename(image_name) + ".png"
		print("Saved file as: " + new_image_path)

		cv2.imwrite(new_image_path, image)


main()
