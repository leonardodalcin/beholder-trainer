import logging as log

# filename='app.log', filemode='w',
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S')
import cv2


class Camera():
    __instance = None
    piCamera = None
    isPreviewing = False

    def takePhoto(self):
        return '/media/leonardo/Images/datasets/production_angle/01_pieces_on_mold/dataset_01/single_piece_position_14/16_58_27_910.pgm'
        # rawCapture = PiRGBArray(self.piCamera)
        # self.piCamera.capture(rawCapture, format="bgr")
        # return rawCapture.array

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Camera.__instance == None:
            Camera()
        return Camera.__instance

    def __init__(self):
        pass
        # print("Initializing Camera")
        # if Camera.__instance != None:
        #     print("Camera was already initialized, throwing exception")
        #     raise Exception("This class is a singleton!")
        # else:
        #     print("Setting PiCamera wrapper")
        #     self.piCamera = PiCamera()
        #     self.piCamera.rotation = 180
