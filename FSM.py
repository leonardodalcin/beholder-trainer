import logging as log
from IO import IO
# filename='app.log', filemode='w',
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S')

Ports =	{
  "open_signal": 2,
  "close_signal": 3
}

from transitions import Machine

class FSM(object):

    state = None
    states = ['injecting', 'testingQuality', 'testingExtraction']
    initial_state = states[0]

    def testQuality(self):
        log.info("Testing quality")
        openMouldSignal = IO(0)
        openMouldSignal.wait_signal()
        return True
    def testExtraction(self):
        log.info("Testing extraction")

        return True
    def inject(self):
        log.info("Injecting")
        return True

    def __init__(self):
        # No anonymous superheroes on my watch! Every narcoleptic superhero gets
        # a name. Any name at all. SleepyMan. SlumberGirl. You get the idea.

        # What have we accomplished today?
        self.kittens_rescued = 0

        # Initialize the state machine
        self.machine = Machine(model=self, states=FSM.states, initial=self.initial_state)

        # Add some transitions. We could also define these using a static list of
        # dictionaries, as we did with states above, and then pass the list to
        # the Machine initializer as the transitions= argument.

        # At some point, every superhero must rise and shine.
        self.machine.add_transition(trigger='testQuality', source='injecting', dest='testingQuality')
        self.machine.add_transition(trigger='testExtraction', source='testingQuality', dest='testingExtraction')
        self.machine.add_transition(trigger='inject', source='testingExtraction', dest='injecting')

        # Useful examples using before, after and conditions to states.
        # # Those calories won't replenish themselves!
        # self.machine.add_transition('eat', 'hungry', 'hanging out')
        #
        # # Superheroes are always on call. ALWAYS. But they're not always
        # # dressed in work-appropriate clothing.
        # self.machine.add_transition('distress_call', '*', 'saving the world',
        #                             before='change_into_super_secret_costume')
        #
        # # When they get off work, they're all sweaty and disgusting. But before
        # # they do anything else, they have to meticulously log their latest
        # # escapades. Because the legal department says so.
        # self.machine.add_transition('complete_mission', 'saving the world', 'sweaty',
        #                             after='update_journal')
        #
        # # Sweat is a disorder that can be remedied with water.
        # # Unless you've had a particularly long day, in which case... bed time!
        # self.machine.add_transition('clean_up', 'sweaty', 'asleep', conditions=['is_exhausted'])
        # self.machine.add_transition('clean_up', 'sweaty', 'hanging out')
        #
        # # Our NarcolepticSuperhero can fall asleep at pretty much any time.
        # self.machine.add_transition('nap', '*', 'asleep')

