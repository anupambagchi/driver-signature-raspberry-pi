#!/usr/bin/env python

from __future__ import with_statement

import getpass
import json
import os
import serial
import signal
import sqlite3
import sys
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from gps import *
from math import pi
from pygsm import GsmModem

import obd_io
import obd_sensors
from obd_utils import scanRadioComm, scanUSBSerial

TIRE_DIAMETER = 22

# 1" = .0254m
INCH_TO_METER = .0254

def get_circumference(diameter_inches):
    diameter_meters = diameter_inches * INCH_TO_METER

    return diameter_meters * pi

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


gpsd = None  # setting the global variable


# os.system('clear')  # clear the terminal (optional)

class GpsPoller(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        global gpsd  # bring it in scope
        gpsd = gps(mode=WATCH_ENABLE)  # starting the stream of info
        self.current_value = None
        self.running = True  # setting the thread running to true

    def run(self):
        global gpsd
        while gps_poller.running:
            gpsd.next()  # this will continue to loop and grab EACH set of gpsd info to clear the buffer


class OBD_Recorder():
    def __init__(self, log_items):
        self.port = None
        self.sensorlist = []
        print("Logging following items:")
        for item in log_items:
            self.add_log_item(item)

        self.gear_ratios = [34 / 13, 39 / 21, 36 / 23, 27 / 20, 26 / 21, 25 / 22]

    def connect(self, portnames):
        # portnames = scanSerial()
        print (portnames)
        for port in portnames:
            self.port = obd_io.OBDPort(port, None, 2, 2)
            if (self.port.State == 0):
                self.port.close()
                self.port = None
            else:
                break

        if (self.port):
            print ("Connected to " + self.port.port.name)

    def is_connected(self):
        return self.port

    def add_log_item(self, item):
        for index, e in enumerate(obd_sensors.SENSORS):
            if (item == e.shortname):
                self.sensorlist.append(index)
                print ("	 " + e.name)
                break

    def get_obd_data(self):
        results = {}
        for index in self.sensorlist:
            (name, value, unit) = self.port.sensor(index)
            results[obd_sensors.SENSORS[index].shortname] = value

        gear = self.calculate_gear(results["rpm"], results["speed"])
        results["gear"] = gear

        return results

    def get_dtc_codes(self):
        return self.port.get_dtc()

    def clear_dtc_codes(self):
        return self.port.clear_dtc()

    def calculate_gear(self, rpm, speed):
        if speed == "" or speed == 0:
            return 0
        if rpm == "" or rpm == 0:
            return 0

        try:
            rps = rpm / 60
            mps = (speed * 1.609 * 1000) / 3600

            primary_gear = 85 / 46  # street triple
            final_drive = 47 / 16

            tyre_circumference = get_circumference(TIRE_DIAMETER)  # meters for 2005 Toyota Highlander

            current_gear_ratio = (rps * tyre_circumference) / (mps * primary_gear * final_drive)

            # print current_gear_ratio
            gear = min((abs(current_gear_ratio - i), i) for i in self.gear_ratios)[1]
            return gear
        except:
            return 0


# Beginning of main program
if __name__ == '__main__':
    runningAs = getpass.getuser()
    if runningAs != 'root':
        print("You need to run this program as root user.")
        exit(-1)

    if len(sys.argv) < 3:
        print("You have to pass the username as a program argument")
        print("Usage: " + sys.argv[0] + " <username> <vehiclemake-model-year>")
        exit(-1)

    username = sys.argv[1]
    vehicle = sys.argv[2]
    # Find out how many USB devices exist

    allRFCommDevicePorts = scanRadioComm()
    allUSBDevicePorts = scanUSBSerial()
    print("RFPorts detected with devices on them: " + str(allRFCommDevicePorts))
    print("USBPorts detected with devices on them: " + str(allUSBDevicePorts))

    usbPortsIdentified = {}

    iccid = ''  # Default values are blank for those that come from GSM modem
    imei = ''

    for usbPort in allUSBDevicePorts:
        try:
            with time_limit(4):
                print ("Trying to connect as GSM to " + str(usbPort))
                gsm = GsmModem(port=usbPort, logger=GsmModem.debug_logger).boot()
                print ("GSM modem detected at " + str(usbPort))
                allUSBDevicePorts.remove(usbPort)  # We just found it engaged, don't use it again
                iccid = gsm.query("AT^ICCID?", "^ICCID:").strip('"')
                imei = gsm.query("ATI", "IMEI:")
                usbPortsIdentified[str(usbPort)] = "gsm"
                print(usbPort, usbPortsIdentified[usbPort])
                break  # We got a port, so break out of loop
        except TimeoutException:
            # Maybe this is not the right port for the GSM modem, so skip to the next number
            print ("Timed out!")
        except IOError:
            print ("IOError - so " + usbPort + " is also not a GSM device")

    # Open a SQLlite3 connection
    dbconnection = sqlite3.connect('/opt/driver-signature-raspberry-pi/database/obd2data.db')
    dbcursor = dbconnection.cursor()

    # Do some cleanup as soon as you start. This is to prevent the database size from growing too big.
    localtime = datetime.now()
    delta = timedelta(days=15)
    fifteendaysago = localtime - delta
    fifteendaysago_str = fifteendaysago.isoformat()
    dbcursor.execute('DELETE FROM CAR_READINGS WHERE EVENTTIME < ?', (fifteendaysago_str,))
    dbconnection.commit()
    dbcursor.execute('VACUUM CAR_READINGS')
    dbconnection.commit()

    gps_poller = GpsPoller()  # create the thread
    try:
        # Initialize the GPS poller
        gps_poller.start()  # start it up
        logitems_full = ["dtc_status", "dtc_ff", "fuel_status", "load", "temp", "short_term_fuel_trim_1",
                         "long_term_fuel_trim_1", "short_term_fuel_trim_2", "long_term_fuel_trim_2",
                         "fuel_pressure", "manifold_pressure", "rpm", "speed", "timing_advance", "intake_air_temp",
                         "maf", "throttle_pos", "secondary_air_status", "o211", "o212", "obd_standard",
                         "o2_sensor_position_b", "aux_input", "engine_time", "abs_load", "rel_throttle_pos",
                         "ambient_air_temp", "abs_throttle_pos_b", "acc_pedal_pos_d", "acc_pedal_pos_e",
                         "comm_throttle_ac", "rel_acc_pedal_pos", "eng_fuel_rate", "drv_demand_eng_torq",
                         "act_eng_torq", "eng_ref_torq"]

        logitems_partial = ["fuel_status", "load", "temp", "short_term_fuel_trim_1",
                            "long_term_fuel_trim_1", "short_term_fuel_trim_2", "long_term_fuel_trim_2",
                            "rpm", "speed", "timing_advance", "intake_air_temp",
                            "maf", "throttle_pos", "o212"]

        # Initialize the OBD recorder
        obd_recorder = OBD_Recorder(logitems_full)
        need_to_exit = False
        try:
            obd_recorder.connect(allRFCommDevicePorts + allUSBDevicePorts)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            print ("Unable to connect to OBD port. Exiting...")
            need_to_exit = True

        if not obd_recorder.is_connected():
            print ("OBD device is not connected. Exiting.")
            need_to_exit = True

        if need_to_exit:
            os._exit(-1)

        # Everything looks good - so start recording
        print ("Database logging started...")
        print ("Ids of records inserted will be printed on screen.")

        lastminute = -1
        need_to_exit = False
        while True:
            # It may take a second or two to get good data
            # print gpsd.fix.latitude,', ',gpsd.fix.longitude,'  Time: ',gpsd.utc
            if need_to_exit:
                os._exit(-1)

            if (obd_recorder.port is None):
                print("Your OBD port has not been set correctly, found None.")
                sys.exit(-1)

            localtime = datetime.now()
            results = obd_recorder.get_obd_data()

            currentminute = localtime.minute
            if currentminute != lastminute:
                dtc_codes = obd_recorder.get_dtc_codes()
                print ('DTC=', str(dtc_codes))
                results["dtc_code"] = dtc_codes
                lastminute = currentminute

            results["username"] = username
            results["vehicle"] = vehicle
            results["eventtime"] = datetime.utcnow().isoformat()
            results["iccid"] = iccid
            results["imei"] = imei

            loc = {}
            loc["type"] = "Point"
            loc["coordinates"] = [gpsd.fix.longitude, gpsd.fix.latitude]
            results["location"] = loc
            results["heading"] = gpsd.fix.track
            results["altitude"] = gpsd.fix.altitude
            results["climb"] = gpsd.fix.climb
            results["gps_speed"] = gpsd.fix.speed
            results["heading"] = gpsd.fix.track

            results_str = json.dumps(results)
            # print(results_str)

            # Insert a row of data
            dbcursor.execute('INSERT INTO CAR_READINGS(EVENTTIME, DEVICEDATA) VALUES (?,?)',
                             (results["eventtime"], results_str))

            # Save (commit) the changes
            dbconnection.commit()

            post_id = dbcursor.lastrowid
            print(post_id)

        # print 'latitude	', gpsd.fix.latitude
        # print 'longitude   ', gpsd.fix.longitude
        # print 'time utc	', gpsd.utc, ' + ', gpsd.fix.time
        # print 'altitude (m)', gpsd.fix.altitude
        # print 'eps		 ', gpsd.fix.eps
        # print 'epx		 ', gpsd.fix.epx
        # print 'epv		 ', gpsd.fix.epv
        # print 'ept		 ', gpsd.fix.ept
        # print 'speed (m/s) ', gpsd.fix.speed
        # print 'climb	   ', gpsd.fix.climb
        # print 'track	   ', gpsd.fix.track
        # print 'mode		', gpsd.fix.mode
        # print 'sats		', gpsd.satellites

    except (KeyboardInterrupt, SystemExit, SyntaxError):  # when you press ctrl+c
        print ("Manual intervention Killing Thread..." + sys.exc_info()[0])
        need_to_exit = True
    except serial.serialutil.SerialException:
        print("Serial connection error detected - OBD device may not be communicating. Exiting." + sys.exc_info()[0])
        need_to_exit = True
    except IOError:
        print("Input/Output error detected. Exiting." + sys.exc_info()[0])
        need_to_exit = True
    except:
        print("Unexpected exception encountered. Exiting." + sys.exc_info()[0])
        need_to_exit = True
    finally:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print(sys.exc_info()[1])
        gps_poller.running = False
        gps_poller.join()  # wait for the thread to finish what it's doing
        dbconnection.close()
        print ("Done.\nExiting.")
        sys.exit(0)
