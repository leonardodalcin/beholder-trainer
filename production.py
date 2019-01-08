# Logging should be a class lather on - TODO
import logging as log

# filename='app.log', filemode='w',
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S')

###

# Initializes FSM
from FSM import FSM
fsm = FSM()

# Code smells - any way to do it better? TODO
shouldRun = True
while (shouldRun):
    # Remove this, stupid, but added for debugging - TODO
    shouldRun = False

    if fsm.state == 'injecting':
        log.info("Entering state: Injecting")
        # waitOpenedSignal()
        fsm.testQuality()

    elif fsm.state == 'testingQuality':
        log.info("Entering state: TestingQuality")

        # take photo
        # crop ROIS
        # predict ROIS
        # define external signal if good or bad
        # save rois, photos and predictions
        # do it all fast!
        fsm.testExtraction()
    elif fsm.state == 'testingExtraction':
        log.info("Entering state: TestingExtraction")

        # take photo
        # crop ROIS
        # predict ROIS
        # define external signal if good or bad
        # save rois, photos and predictions
        # do it all fast!
        fsm.inject()
