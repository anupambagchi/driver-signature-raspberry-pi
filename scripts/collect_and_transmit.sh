#!/usr/bin/env bash

runningprocs=`ps auxww | grep obd_transmitter.py | grep -v grep | wc -l`
if [ "$runningprocs" != "0" ] ; then
    # echo "A process is already running on this machine"
    exit
fi
cd /opt/driver-signature-raspberry-pi/src
python /opt/driver-signature-raspberry-pi/src/obd_transmitter.py