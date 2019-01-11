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

    def __init__(self):
        self.camera = Camera()
        # No anonymous superheroes on my watch! Every narcoleptic superhero gets
        # a name. Any name at all. SleepyMan. SlumberGirl. You get the idea.

        # What have we accomplished today?
        self.kittens_rescued = 0

        # Initialize the state machine
        self.machine = Machine(model=self, states=self.states, initial=self.initial_state)
        self.machine.add_ordered_transitions()
        self.machine.on_enter_injecting('inject')
        self.machine.on_enter_testing_quality('test_quality')
        self.machine.on_enter_testing_extraction('test_extraction')

        # Add some transitions. We could also define these using a static list of
        # dictionaries, as we did with states above, and then pass the list to
        # the Machine initializer as the transitions= argument.

        # At some point, every superhero must rise and shine.

        # See the "alternative initialization" section for an explanation of the 1st argument to init
    def test_quality(self):
        log.info("Testing quality")
        open_mould_signal = IO(Ports["open_signal"])
        log.info("Waiting for open signal")
        open_mould_signal.wait_signal()
        log.info("Open signal acknowledged")
        log.info("Taking photo")
        test_quality_image = Image(path = self.camera.takePhoto())
        test_quality_image.show()



    def test_extraction(self):
        log.info("Testing extraction")
        extration_complete_signal = IO(Ports["extraction_signal"])
        log.info("Waiting for extraction signal")
        extration_complete_signal.wait_signal()

    def inject(self):
        log.info("Injecting")
        # organize photos?
