#!/usr/bin/env bash

runningprocs=`ps auxww | grep obd_transmitter.py | grep -v grep | wc -l`
if [ "$runningprocs" != "0" ] ; then
    # echo "A process is already running on this machine"
    exit
fi
cd /opt/obd-edge-computing/src
python /opt/obd-edge-computing/src/obd_transmitter.py