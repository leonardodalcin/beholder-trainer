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
    fsm.next_state()