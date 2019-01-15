import json
import cv2
import glob
import os
import uuid
from Image import Image

root_path = "/media/leonardo/Images/full_mould_bad"
def main():
    for file in glob.glob(root_path + "/*.json", recursive=True):
        with open(file) as json_data:
            json_file = json.load(json_data)
            print(json_file)
            for image_name in glob.glob(str(json_file["image_path"]) + "/*.pgm"):
                image = cv2.imread(image_name, 0)

                for roi in json_file["coords"]:

                    y1 = roi["y1"]
                    y2 = roi["y2"]
                    x1 = roi["x1"]
                    x2 = roi["x2"]
                    if y1 > y2:
                        temp = y1
                        y1 = y2
                        y2 = temp
                    if x1 > x2:
                        temp = x1
                        x1 = x2
                        x2 = temp

                    cropped_image = image[y1:y2, x1:x2]
                    img = Image(cropped_image)
                    if(y1 > 2500):
                        cv2.flip(img.image, 0, img.image)

                        # cv2.flip(img.image, 1, img.image)
                        img.show()

                    cropped_image = img.image
                    for image_class in json_file["classes"]:
                        save_path = root_path + "/classes/" + image_class
                        if (not os.path.exists(save_path)):
                            os.makedirs(save_path)
                        image_name = save_path + "/" + str(uuid.uuid4()) + image_class + ".jpg"
                        try:
                            cv2.imwrite(image_name, cropped_image, [cv2.IMWRITE_PXM_BINARY])
                        except:
                            print("Error writing")
                        print("Saving image to " + image_name)



main()
