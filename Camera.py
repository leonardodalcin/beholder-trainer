import logging as log

# filename='app.log', filemode='w',
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S')
import cv2
import PySpin
from Image import Image
import numpy as np
import time

class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2

CHOSEN_TRIGGER = TriggerType.SOFTWARE


def configure_trigger(cam):
    """
    This function configures the camera to use a trigger. First, trigger mode is
    set to off in order to select the trigger source. Once the trigger source
    has been selected, trigger mode is then enabled, which has the camera
    capture only a single image upon the execution of the chosen trigger.

     :param cam: Camera to configure trigger for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """
    result = True

    print('*** CONFIGURING TRIGGER ***\n')

    if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
        print('Software trigger chosen ...')
    elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
        print('Hardware trigger chose ...')

    try:
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        nodemap = cam.GetNodeMap()
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        print('Trigger mode disabled...')

        # Select trigger source
        # The trigger source must be set to hardware or software while trigger
        # mode is off.
        node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
            print('Unable to get trigger source (node retrieval). Aborting...')
            return False

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
            if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
                    node_trigger_source_software):
                print('Unable to set trigger source (enum entry retrieval). Aborting...')
                return False
            node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())

        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            node_trigger_source_hardware = node_trigger_source.GetEntryByName('Line0')
            if not PySpin.IsAvailable(node_trigger_source_hardware) or not PySpin.IsReadable(
                    node_trigger_source_hardware):
                print('Unable to set trigger source (enum entry retrieval). Aborting...')
                return False
            node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())

        # Turn trigger mode on
        # Once the appropriate trigger source has been set, turn trigger mode
        # on in order to retrieve images using the trigger.
        node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
        if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
            print('Unable to enable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
        print('Trigger mode turned back on...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def grab_next_image_by_trigger(nodemap):
    """
    This function acquires an image by executing the trigger node.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    # Use trigger to capture image
    # The software trigger only feigns being executed by the Enter key;
    # what might not be immediately apparent is that there is not a
    # continuous stream of images being captured; in other examples that
    # acquire images, the camera captures a continuous stream of images.
    # When an image is retrieved, it is plucked from the stream.

    # Execute software trigger
    node_softwaretrigger_cmd = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
    if not PySpin.IsAvailable(node_softwaretrigger_cmd) or not PySpin.IsWritable(node_softwaretrigger_cmd):
        print('Unable to execute trigger. Aborting...')
        return False

    node_softwaretrigger_cmd.Execute()

    # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger


class Camera():
    camera = None
    nodemap = None
    nodemap_tldevice = None

    def take_photo(self):

        grab_next_image_by_trigger(self.nodemap)

        image_result = self.camera.GetNextImage()
        if image_result.IsIncomplete():
            print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

        width = image_result.GetWidth()
        height = image_result.GetHeight()

        img = Image(cv2.cvtColor(np.array(image_result.GetData(), dtype="uint8").reshape((height, width)),cv2.COLOR_BAYER_BG2GRAY))
        image_result.Release()
        return img

    def __init__(self):
        """
           Example entry point; please see Enumeration example for more in-depth
           comments on preparing and cleaning up the system.

           :return: True if successful, False otherwise.
           :rtype: bool
           """
        self.system = PySpin.System.GetInstance()
        # Retrieve list of cameras from the system
        cam_list = self.system.GetCameras()
        self.camera = cam_list.GetByIndex(0)

        self.camera.Init()
        self.nodemap = self.camera.GetNodeMap()
        configure_trigger(self.camera)

        node_acquisition_mode = PySpin.CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to SingleFrame (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_single_frame = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_single_frame) or not PySpin.IsReadable(
                node_acquisition_mode_single_frame):
            print('Unable to set acquisition mode to SingleFrame (entry retrieval). Aborting...')
            return False

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(node_acquisition_mode_single_frame.GetValue())

        print('Acquisition mode set to continuous...')
        self.camera.BeginAcquisition()



