#!/usr/bin/env python

# This program collects the data accumulated for the past one minute, summarizes it and transmits it over to the cloud.
# Since the data collection is local and thus assumed to be instantaneous, this program wakes up every minute
# and does a summarization of the past minute, and keeps the transmission packet ready in its database. Then it attempts
# to look at all past summaries which have not been transmitted yet, and attempts to transmit them to the cloud. Any failure
# of transmission is ignored. The un-transmitted records are attempted next time when the program wakes up again. Over time
# all data is transmitted even if it fails a few times in between.

# This program will be invoked once a minute through a cron job.

import base64
import json
import numpy
import requests
import sqlite3
import time
import zlib
from Crypto import Random
from Crypto.PublicKey import RSA
from datetime import datetime
from datetime import timedelta
from itertools import chain
from random import randint
from scipy import stats

from AESCipher import AESCipher
from PKS1_OAEPCipher import PKS1_OAEPCipher


class DataCompactor():
    @staticmethod
    def collect():
        # First find out the id of the record that was included in the last compaction task
        dbconnection = sqlite3.connect('/opt/driver-signature-raspberry-pi/database/obd2data.db')
        dbcursor = dbconnection.cursor()
        last_id_found = dbcursor.execute('SELECT LAST_PROCESSED_ID FROM LAST_PROCESSED WHERE TABLE_NAME = "CAR_READINGS" LIMIT 1')

        lastId = 0
        try:
            first_row = next(last_id_found)
            for row in chain((first_row,), last_id_found):
                pass  # do something
                lastId = row[0]
        except StopIteration as e:
            pass  # 0 results

        # Collect data till the last minute last second, but not including the current minute
        nowTime = datetime.utcnow().isoformat()  # Example: 2017-05-14T19:51:29.071710 in ISO 8601 extended format
        # nowTime = '2017-05-14T19:54:58.398073'  # for testing
        timeTillLastMinuteStr = nowTime[:17] + "00.000000"
        # timeTillLastMinute = dateutil.parser.parse(timeTillLastMinuteStr) # ISO 8601 extended format

        dbcursor.execute('SELECT * FROM CAR_READINGS WHERE ID > ? AND EVENTTIME <= ?', (lastId,timeTillLastMinuteStr))

        allRecords = []
        finalId = lastId
        for row in dbcursor:
            record = row[2]
            allRecords.append(json.loads(record))
            finalId = row[0]

        if lastId == 0:
            # print("Inserting")
            dbcursor.execute('INSERT INTO LAST_PROCESSED (TABLE_NAME, LAST_PROCESSED_ID) VALUES (?,?)', ("CAR_READINGS", finalId))
        else:
            # print("Updating")
            dbcursor.execute('UPDATE LAST_PROCESSED SET LAST_PROCESSED_ID = ? WHERE TABLE_NAME = "CAR_READINGS"', (finalId,))

        #print allRecords
        dbconnection.commit()   # Save (commit) the changes
        dbconnection.close()  # And close it before exiting
        print("Collecting all records till %s comprising IDs from %d to %d ..." % (timeTillLastMinuteStr, lastId, finalId))

        encryptionKeyHandle = open('encryption.key', 'r')
        encryptionKey = RSA.importKey(encryptionKeyHandle.read())
        encryptionKeyHandle.close()

        # From here we need to break down the data into chunks of each minute and store one record for each minute
        minutePackets = {}
        for record in allRecords:
            eventTimeByMinute = record["eventtime"][:17] + "00.000000"
            if eventTimeByMinute in minutePackets:
                minutePackets[eventTimeByMinute].append(record)
            else:
                minutePackets[eventTimeByMinute] = [record]

        # print (minutePackets)
        summarizationItems = ['load', 'rpm', 'timing_advance', 'speed', 'altitude', 'gear', 'intake_air_temp',
                              'gps_speed', 'short_term_fuel_trim_2', 'o212', 'short_term_fuel_trim_1', 'maf',
                              'throttle_pos', 'climb', 'temp', 'long_term_fuel_trim_1', 'heading', 'long_term_fuel_trim_2']

        dbconnection = sqlite3.connect('/opt/driver-signature-raspberry-pi/database/obd2summarydata.db')
        dbcursor = dbconnection.cursor()
        for minuteStamp in minutePackets:
            minutePack = minutePackets[minuteStamp]
            packet = {}
            packet["timestamp"] = minuteStamp
            packet["data"] = minutePack
            packet["summary"] = DataCompactor.summarize(minutePack, summarizationItems)

            packetStr = json.dumps(packet)

            # Create an AES encryptor
            aesCipherForEncryption = AESCipher()
            symmetricKey = Random.get_random_bytes(32)   # generate a random key
            aesCipherForEncryption.setKey(symmetricKey)  # and set it within the encryptor
            encryptedPacketStr = aesCipherForEncryption.encrypt(packetStr)

            # Compress the packet
            compressedPacket = base64.b64encode(zlib.compress(encryptedPacketStr))  # Can be transmitted
            dataSize = len(packetStr)

            # Now do asymmetric encryption of the key using PKS1_OAEP
            pks1OAEPForEncryption = PKS1_OAEPCipher()
            pks1OAEPForEncryption.readEncryptionKey('encryption.key')
            symmetricKeyEncrypted = base64.b64encode(pks1OAEPForEncryption.encrypt(symmetricKey))  # Can be transmitted

            dbcursor.execute('INSERT INTO PROCESSED_READINGS(EVENTTIME, DEVICEDATA, ENCKEY, DATASIZE) VALUES (?,?,?,?)',
                             (minuteStamp, compressedPacket, symmetricKeyEncrypted, dataSize))

        # Save this list to another table
        dbconnection.commit()   # Save (commit) the changes
        dbconnection.close()  # And close it before exiting

    '''
    This function takes an array of readings (which individually is a dictionary converted from JSON) and summarizes all
    values found in items. The summary contains common statistical aggregations for the values found in this packet.
    '''
    @staticmethod
    def summarize(readings, items):
        summary = {}
        for item in items:
            summaryItem = {}
            itemarray = []
            for reading in readings:
                if isinstance(reading[item], (float, int)):
                    itemarray.append(reading[item])
            # print(itemarray)
            summaryItem["count"] = len(itemarray)
            if len(itemarray) > 0:
                summaryItem["mean"] = numpy.mean(itemarray)
                summaryItem["median"] = numpy.median(itemarray)
                summaryItem["mode"] = stats.mode(itemarray)[0][0]
                summaryItem["stdev"] = numpy.std(itemarray)
                summaryItem["variance"] = numpy.var(itemarray)
                summaryItem["max"] = numpy.max(itemarray)
                summaryItem["min"] = numpy.min(itemarray)

            summary[item] = summaryItem

        #print(summary)
        return summary

    @staticmethod
    def transmit():
        base_url = "http://OBD-EDGE-DATA-CATCHER-1969985272.us-west-2.elb.amazonaws.com"   # for accessing it from outside the firewall

        url = base_url + "/obd2/api/v1/17350/upload"

        dbconnection = sqlite3.connect('/opt/driver-signature-raspberry-pi/database/obd2summarydata.db')
        dbcursor = dbconnection.cursor()
        dbupdatecursor = dbconnection.cursor()

        dbcursor.execute('SELECT ID, EVENTTIME, TRANSMITTED, DEVICEDATA, ENCKEY, DATASIZE FROM PROCESSED_READINGS WHERE TRANSMITTED="FALSE" ORDER BY EVENTTIME')
        for row in dbcursor:
            rowid = row[0]
            eventtime = row[1]
            devicedata = row[3]
            enckey = row[4]
            datasize = row[5]

            payload = {'size': str(datasize), 'key': enckey, 'data': devicedata, 'eventtime': eventtime}
            response = requests.post(url, json=payload)

            #print(response.text)  # TEXT/HTML
            #print(response.status_code, response.reason)  # HTTP

            if response.status_code == 201:
                dbupdatecursor.execute('UPDATE PROCESSED_READINGS SET TRANSMITTED="TRUE" WHERE ID = ?', (rowid,))
                dbconnection.commit()  # Save (commit) the changes

        dbconnection.commit()   # Save (commit) the changes
        dbconnection.close()  # And close it before exiting

    @staticmethod
    def cleanup():
        localtime = datetime.now()
        if int(localtime.isoformat()[14:16]) == 0:
            delta = timedelta(days=15)
            fifteendaysago = localtime - delta
            fifteendaysago_str = fifteendaysago.isoformat()
            dbconnection = sqlite3.connect('/opt/driver-signature-raspberry-pi/database/obd2summarydata.db')
            dbcursor = dbconnection.cursor()
            dbcursor.execute('DELETE FROM PROCESSED_READINGS WHERE EVENTTIME < ?', (fifteendaysago_str,))
            dbconnection.commit()
            dbcursor.execute('VACUUM PROCESSED_READINGS')

            dbconnection.commit()   # Save (commit) the changes
            dbconnection.close()  # And close it before exiting

# Beginning of main program
if __name__ == '__main__':
    with_wait_time = False  # Turn this to False if you do not want to stagger the transmission. True in production.

    # Wait for a while to allow all data for the past minute to be collected in the database first
    if with_wait_time:
        time.sleep(10)
    DataCompactor.collect()

    # We do not want the server to be pounded with requests all at the same time
    # So we have a random wait time to distribute it over the next 30 seconds.
    # This brings the max wait time per minute to be 40 seconds, which is still 20 seconds to do the job (summarize + transmit).
    waitminutes = randint(0, 30)
    if with_wait_time:
        time.sleep(waitminutes)
    DataCompactor.transmit()
    DataCompactor.cleanup()

