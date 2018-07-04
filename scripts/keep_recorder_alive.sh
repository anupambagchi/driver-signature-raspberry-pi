#!/usr/bin/env bash

runningprocs=`ps auxww | grep obd_sqlite_recorder.py | grep -v grep | wc -l`
if [ "$runningprocs" == "0" ] ; then
    iam=`whoami`
    cd /opt/obd-edge-computing/src
    python /opt/obd-edge-computing/src/obd_sqlite_recorder.py $iam
fi
