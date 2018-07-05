#!/usr/bin/env bash

runningprocs=`ps auxww | grep obd_sqlite_recorder.py | grep -v grep | wc -l`
if [ "$runningprocs" == "0" ] ; then
    iam=`whoami`
    cd /opt/driver-signature-raspberry-pi/src
    python /opt/driver-signature-raspberry-pi/src/obd_sqlite_recorder.py $iam
fi
