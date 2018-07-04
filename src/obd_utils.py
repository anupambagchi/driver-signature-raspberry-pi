import serial
import subprocess
from time import sleep

def scanSerial():
    """scan for available ports. return a list of serial names"""
    available = []
    # Enable Bluetooh connection
    for i in range(10):
        try:
            s = serial.Serial("/dev/rfcomm" + str(i))
            available.append((str(s.port)))
            s.close()  # explicit close 'cause of delayed GC in java
        except serial.SerialException:
            pass
        # Enable USB connection
    #for i in range(1, 256):
    #    try:
    #        # print i
    #        s = serial.Serial("/dev/ttyUSB" + str(i))
    #        available.append(s.portstr)
    #        s.close()  # explicit close 'cause of delayed GC in java
    #    except serial.SerialException:
    #        pass
    # # Enable obdsim
    #		 # for i in range(256):
    #		 # try: #scan Simulator
    #		 # s = serial.Serial("/dev/pts/"+str(i))
    #		 # available.append(s.portstr)
    #		 # s.close()   # explicit close 'cause of delayed GC in java
    #		 # except serial.SerialException:
    #		 # pass
    print("Found the following ports available: " + str(available))
    return available


def scanRadioComm():
    """scan for available ports. return a list of serial names"""
    available = []
    rfPortsDetected = {}
    try:
        for line in runCommand("ls -C1 /dev/rfcomm*"):
            rfPortsDetected[line.rstrip()] = ''

        # Enable Bluetooth connection
        for rfPort, v in rfPortsDetected.items():
            try:
                s = serial.Serial(str(rfPort))
                available.append((str(s.port)))
                s.close()  # explicit close 'cause of delayed GC in java
            except serial.SerialException:
                pass
            # Enable USB connection
    except:
        pass

    return available


def scanUSBSerial():
    """scan for available ports. return a list of serial names"""
    available = []
    usbPortsDetected = {}
    for line in runCommand("ls -C1 /dev/ttyUSB*"):
        usbPortsDetected[line.rstrip()] = ''

    for usbPort, v in usbPortsDetected.items():
        try:
            # print "Trying " + str(usbPort)
            s = serial.Serial(str(usbPort))
            available.append(str(s.portstr))
            s.close()  # explicit close 'cause of delayed GC in java
        except serial.SerialException:
            pass
        except IOError:
            pass

    return available


def runCommand(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # Read stdout from subprocess until the buffer is empty !
    for line in iter(p.stdout.readline, b''):
        if line:  # Don't print blank lines
            yield line
    # This ensures the process has completed, AND sets the 'returncode' attr
    while p.poll() is None:
        sleep(.1)  # Don't waste CPU-cycles
    # Empty STDERR buffer
    err = p.stderr.read()
    if p.returncode != 0:
        # The run_command() function is responsible for logging STDERR
        print("Error: " + err)
