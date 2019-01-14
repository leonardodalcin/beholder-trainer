import logging as log
from IO import IO
from Camera import Camera
from Image import Image
from DNN import Predictor
# filename='app.log', filemode='w',
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S')

Ports =	{
  "open_signal": 0,
  "extraction_signal": 1,
  "close_signal": 2
}

from transitions import Machine

class FSM(object):

    state = None
    states = ['injecting', 'testing_quality', 'testing_extraction']
    initial_state = states[0]
    camera = None


    def __init__(self):
        self.camera = Camera()
        # Initialize the state machine
        self.machine = Machine(model=self, states=self.states, initial=self.initial_state)
        self.machine.add_ordered_transitions()
        self.machine.on_enter_injecting('inject')
        self.machine.on_enter_testing_quality('test_quality')
        self.machine.on_enter_testing_extraction('test_extraction')

    def test_quality(self):
        log.info("Testing quality")
        log.info("Waiting for open signal")
        open_mould_signal = IO(Ports["open_signal"])
        open_mould_signal.wait_signal()
        log.info("Open signal acknowledged")
        log.info("Taking photo")
        test_quality_image = self.camera.take_photo()
        test_quality_image.save("full")


    def test_extraction(self):
        log.info("Testing extraction")

        log.info("Waiting for extraction signal")
        extration_complete_signal = IO(Ports["extraction_signal"])
        extration_complete_signal.wait_signal()
        log.info("Open signal acknowledged")
        log.info("Taking photo")
        test_extraction_image = self.camera.take_photo()
        test_extraction_image.save("empty")

    def inject(self):
        log.info("Injecting")
        # organize photos?
