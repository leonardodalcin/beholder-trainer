# License
# -------
# This code is published and shared by Numato Systems Pvt Ltd under GNU LGPL
# license with the hope that it may be useful. Read complete license at
# http://www.gnu.org/licenses/lgpl.html or write to Free Software Foundation,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

# Simplicity and understandability is the primary philosophy followed while
# writing this code. Sometimes at the expence of standard coding practices and
# best practices. It is your responsibility to independantly assess and implement
# coding practices that will satisfy safety and security necessary for your final
# application.

# This demo code demonstrates how to read the status of a GPIO


import sys
import serial

class IO(object):
    port_name = "/dev/ttyACM0"
    gpio_number = None
    serial_port = None

    def __init__(self, gpio_number):
        self.gpio_number = gpio_number
        self.serial_port = serial.Serial(self.port_name, 19200, timeout=0.001)

    def clear(self):
        self.serial_port.write(str.encode("gpio clear " + str(self.gpio_number) + "\r"))

    def close_port(self):
        self.serial_port.close()

    def wait_signal(self):
        while (self.read_port() == 0):
            continue
        self.clear()
        self.close_port()

    def read_port(self):
        self.serial_port.write(str.encode("gpio read " + str(self.gpio_number) + "\r"))
        response = self.serial_port.read(25)
        if (response[-4] == 49):
            return 1
        elif (response[-4] == 48):
            return 0