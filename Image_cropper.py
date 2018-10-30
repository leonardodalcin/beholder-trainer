import json
import cv2
import glob



def main():
	for file in glob.glob("/Users/leonardodalcin/Downloads/*.json"):
		with open(file) as json_data:
			json_file = json.load(json_data)
			print(json_file)
			for image_name in glob.glob(str(json_file["image_path"]) + "/*.pgm"):
				image = cv2.imread(image_name)
				for roi in json_file["coords"]:
					y1 =roi["y1"]
					y2 = roi["y2"]
					x1 = roi["x1"]
					x2 = roi["x2"]
					cropped_image = image[y1:y2,x1:x2]
					cv2.imshow("test", cropped_image)
					cv2.waitKey(0)


main()