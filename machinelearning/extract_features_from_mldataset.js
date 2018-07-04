// This script looks at the precomputed_segments table and summarizes values for each segment that might be considered as
// "driving feature sets". These feature sets can be used to determine driver signatures and rank drivers based on their behavior.
//
// mongo --nodb --quiet --eval "var host='localhost';" extract_features_from_mldataset.js

/*
  This script looks up the book-keeping table to find out the time span it needs to process. Then it scans through all
  records and writes all the raw records to another collection (mldataset). While doing so it adds some extra information
  which can be derived from the current record and the previous record of the same driving session. This extra information
  is stored in a special paragraph called 'delta'.
 */
load("GeoDistance.js");

var db = connect(host + ":27017/obd2");
var startFromScratch = true;   // This flag will rebuild all the segment-clusters from scratch

// Look up the book-keeping table to figure out the time from where we need to pick up
// Look up the book-keeping table to figure out the time from where we need to pick up
var processedUntil = db.getCollection('book_keeping').findOne( { _id: "driver_vehicles_processed_until" } );
var currentTime = new Date();  // This time includes the seconds value
// Set the end time (for querying) to the current time till the last whole minute, excluding seconds
var endTimeGlobal = new Date(Date.UTC(currentTime.getFullYear(), currentTime.getMonth(), currentTime.getDate(), currentTime.getHours(), currentTime.getMinutes(), 0, 0))

if (processedUntil === null) {
  db.book_keeping.insert( { _id: "driver_vehicles_processed_until", lastEndTimes: [] } ); // initialize to an empty array
}

// Now do a query of the database to find out what records are new since we ran it last
var startTimeForSearchingActiveDevices = new Date(endTimeGlobal.getTime() - (200*86400000)); // Go back 200 days

/*
var processedUntil = db.getCollection('book_keeping').findOne( { _id: "summary_processed_until" } );
var endTime = new Date();  // Set the end time to the current time
var startTime;
if (processedUntil === null) {
  db.getCollection('book_keeping').insert( { _id: "summary_processed_until", lastEndTime: endTime } );
  startTime = new Date(endTime.getTime() - (365*86400000));  // Go back 365 days
} else {
  startTime = processedUntil.lastEndTime;
}
*/

// Another book-keeping task is to read the driver-vehicle hash-table from the database
// Look up the book-keeping table to figure out the previous driver-vehicle codes (we have
// numbers representing the combination of drivers and vehicles).
var driverVehicles = db.getCollection('book_keeping').findOne( { _id: "driver_vehicles" } );
var drivers;
if (driverVehicles !== null)
  drivers = driverVehicles.drivers;
else
  drivers = {};

var maxDriverVehicleId = 0;
for (var key in drivers) {
  if (drivers.hasOwnProperty(key)) {
    maxDriverVehicleId = Math.max(maxDriverVehicleId, drivers[key]);
  }
}

// Now do a query of the database to find out what records are new since we ran it last
var allNewCarDrivers = db.getCollection('mldataset').aggregate([
  {
    "$match": {
      "data.eventTimestamp" : { $gt: startTimeForSearchingActiveDevices, $lte: endTimeGlobal }
    }
  },
  {
    "$group": { _id: "$data.username" }
  }
]);

// Then look at all the records returned and process each driver one-by-one
// The following query is a pipeline with the following steps:
allNewCarDrivers.forEach(function(driverId) {
  var driverName = driverId._id;
  print("Processing driver: " + driverName);
  var startTimeForDriver = startTimeForSearchingActiveDevices; // To begin with we start with the earliest start time we care about

  var driverIsNew = true;
  // First find out if this device already has some records processed, and has a last end time defined
  var lastEndTimeDevice = db.getCollection('book_keeping').find(
    {
      _id: "driver_vehicles_processed_until",
      "lastEndTimes.driver": driverName
    },
    {
      _id: 0,
      'lastEndTimes.$': 1
    }
  );

  lastEndTimeDevice.forEach(function(record) {
    startTimeForDriver = record.lastEndTimes[0].endTime;
    driverIsNew = false;
  });

  //print('Starting time for driver is ' + startTimeForDriver.toISOString());
  //print('endTimeGlobal = ' + endTimeGlobal.toISOString());

  var allNewCarReadings = db.getCollection('mldataset').aggregate([
      {
        "$match": { // 1. Match all records that fall within the time range we have decided to use. Note that this is being
          // done on a live database - which means that new data is coming in while we are trying to analyze it.
          // Thus we have to pin both the starting time and the ending time. Pinning the endtime to the starting time
          // of the application ensures that we will be accurately picking up only the NEW records when the program
          // runs again the next time.
          "data.eventTimestamp": {$gt: startTimeForDriver, $lte: endTimeGlobal},
          "data.username": driverName  // Only consider the records for this specific driver, ignoring all others.
        }
      },
      {
        $project: {  // We only need to consider a few fields for our analysis. This eliminates the summaries from our analysis.
          "data": 1,
          "account": 1,
          "delta": 1,
          "driverVehicleId": 1,
          "_id": 0
        }
      },
      {
        "$group": {
          "_id": {
            year: {$year: "$data.eventTimestamp"},
            month: {$month: "$data.eventTimestamp"},
            day: {$dayOfMonth: "$data.eventTimestamp"},
            hour: {$hour: "$data.eventTimestamp"},
            minute: {$minute: "$data.eventTimestamp"},
            quarter: {$mod: [{$second: "$data.eventTimestamp"}, 4]}
          },

          "averageGPSLatitude": {$avg: {"$arrayElemAt": ["$data.location.coordinates", 1]}},
          "averageGPSLongitude": {$avg: {"$arrayElemAt": ["$data.location.coordinates", 0]}},

          "averageLoad": {$avg: "$data.load"},
          "minLoad": {$min: "$data.load"},
          "maxLoad": {$max: "$data.load"},

          "averageThrottlePosB": {$avg: "$data.abs_throttle_pos_b"},
          "minThrottlePosB": {$min: "$data.abs_throttle_pos_b"},
          "maxThrottlePosB": {$max: "$data.abs_throttle_pos_b"},

          "averageRpm": {$avg: "$data.rpm"},
          "minRpm": {$min: "$data.rpm"},
          "maxRpm": {$max: "$data.rpm"},

          "averageThrottlePos": {$avg: "$data.throttle_pos"},
          "minThrottlePos": {$min: "$data.throttle_pos"},
          "maxThrottlePos": {$max: "$data.throttle_pos"},

          "averageIntakeAirTemp": {$avg: "$data.intake_air_temp"},
          "minIntakeAirTemp": {$min: "$data.intake_air_temp"},
          "maxIntakeAirTemp": {$max: "$data.intake_air_temp"},

          "averageSpeed": {$avg: "$data.speed"},
          "minSpeed": {$min: "$data.speed"},
          "maxSpeed": {$max: "$data.speed"},

          "averageAltitude": {$avg: "$data.altitude"},
          "minAltitude": {$min: "$data.altitude"},
          "maxAltitude": {$max: "$data.altitude"},

          "averageCommThrottleAc": {$avg: "$data.comm_throttle_ac"},
          "minCommThrottleAc": {$min: "$data.comm_throttle_ac"},
          "maxCommThrottleAc": {$max: "$data.comm_throttle_ac"},

          "averageEngineTime": {$avg: "$data.engine_time"},
          "minEngineTime": {$min: "$data.engine_time"},
          "maxEngineTime": {$max: "$data.engine_time"},

          "averageAbsLoad": {$avg: "$data.abs_load"},
          "minAbsLoad": {$min: "$data.abs_load"},
          "maxAbsLoad": {$max: "$data.abs_load"},

          "averageGear": {$avg: "$data.gear"},
          "minGear": {$min: "$data.gear"},
          "maxGear": {$max: "$data.gear"},

          "averageRelThrottlePos": {$avg: "$data.rel_throttle_pos"},
          "minRelThrottlePos": {$min: "$data.rel_throttle_pos"},
          "maxRelThrottlePos": {$max: "$data.rel_throttle_pos"},

          "averageAccPedalPosE": {$avg: "$data.acc_pedal_pos_e"},
          "minAccPedalPosE": {$min: "$data.acc_pedal_pos_e"},
          "maxAccPedalPosE": {$max: "$data.acc_pedal_pos_e"},

          "averageAccPedalPosD": {$avg: "$data.acc_pedal_pos_d"},
          "minAccPedalPosD": {$min: "$data.acc_pedal_pos_d"},
          "maxAccPedalPosD": {$max: "$data.acc_pedal_pos_d"},

          "averageGpsSpeed": {$avg: "$data.gps_speed"},
          "minGpsSpeed": {$min: "$data.gps_speed"},
          "maxGpsSpeed": {$max: "$data.gps_speed"},

          "averageShortTermFuelTrim2": {$avg: "$data.short_term_fuel_trim_2"},
          "minShortTermFuelTrim2": {$min: "$data.short_term_fuel_trim_2"},
          "maxShortTermFuelTrim2": {$max: "$data.short_term_fuel_trim_2"},

          "averageO211": {$avg: "$data.o211"},
          "minO211": {$min: "$data.o211"},
          "maxO211": {$max: "$data.o211"},

          "averageO212": {$avg: "$data.o212"},
          "minO212": {$min: "$data.o212"},
          "maxO212": {$max: "$data.o212"},

          "averageShortTermFuelTrim1": {$avg: "$data.short_term_fuel_trim_1"},
          "minShortTermFuelTrim1": {$min: "$data.short_term_fuel_trim_1"},
          "maxShortTermFuelTrim1": {$max: "$data.short_term_fuel_trim_1"},

          "averageMaf": {$avg: "$data.maf"},
          "minMaf": {$min: "$data.maf"},
          "maxMaf": {$max: "$data.maf"},

          "averageTimingAdvance": {$avg: "$data.timing_advance"},
          "minTimingAdvance": {$min: "$data.timing_advance"},
          "maxTimingAdvance": {$max: "$data.timing_advance"},

          "averageClimb": {$avg: "$data.climb"},
          "minClimb": {$min: "$data.climb"},
          "maxClimb": {$max: "$data.climb"},

          "averageFuelPressure": {$avg: "$data.fuel_pressure"},
          "minFuelPressure": {$min: "$data.fuel_pressure"},
          "maxFuelPressure": {$max: "$data.fuel_pressure"},

          "averageTemp": {$avg: "$data.temp"},
          "minTemp": {$min: "$data.temp"},
          "maxTemp": {$max: "$data.temp"},

          "averageAmbientAirTemp": {$avg: "$data.ambient_air_temp"},
          "minAmbientAirTemp": {$min: "$data.ambient_air_temp"},
          "maxAmbientAirTemp": {$max: "$data.ambient_air_temp"},

          "averageManifoldPressure": {$avg: "$data.manifold_pressure"},
          "minManifoldPressure": {$min: "$data.manifold_pressure"},
          "maxManifoldPressure": {$max: "$data.manifold_pressure"},

          "averageLongTermFuelTrim1": {$avg: "$data.long_term_fuel_trim_1"},
          "minLongTermFuelTrim1": {$min: "$data.long_term_fuel_trim_1"},
          "maxLongTermFuelTrim1": {$max: "$data.long_term_fuel_trim_1"},

          "averageLongTermFuelTrim2": {$avg: "$data.long_term_fuel_trim_2"},
          "minLongTermFuelTrim2": {$min: "$data.long_term_fuel_trim_2"},
          "maxLongTermFuelTrim2": {$max: "$data.long_term_fuel_trim_2"},

          "averageGPSAcceleration": {$avg: "$delta.acceleration"},
          "minGPSAcceleration": {$min: "$delta.acceleration"},
          "maxGPSAcceleration": {$max: "$delta.acceleration"},

          "averageHeadingChange": {$avg: {$abs: "$delta.angular_velocity"}},
          "minHeadingChange": {$min: {$abs: "$delta.angular_velocity"}},
          "maxHeadingChange": {$max: {$abs: "$delta.angular_velocity"}},

          "averageIncline": {$avg: "$data.incline"},
          "minIncline": {$min: "$data.incline"},
          "maxIncline": {$max: "$data.incline"},

          "averageAcceleration": {$avg: "$delta.acceleration"},
          "minAcceleration": {$min: "$delta.acceleration"},
          "maxAcceleration": {$max: "$delta.acceleration"},

          // "dtcCodes": {"$push": "$data.dtc_status"},
          "accountIdArray": {$addToSet: "$account"},

          "vehicleArray": {$addToSet: "$data.vehicle"},
          "driverArray": {$addToSet: "$data.username"},
          "driverVehicleArray": {$addToSet: "$driverVehicleId"},

          "count": {$sum: 1}
        }
      },
      {
        $sort: {
          "_id": 1  // Finally sort the data based on eventtime in ascending order
        }
      }
    ],
    {
      allowDiskUse: true
    }
  );


  var lastRecordedTimeForDriver = startTimeForDriver;
  var insertCounter = 0;
  allNewCarReadings.forEach(function (record) {
    var currentRecordEventTime = new Date(Date.UTC(record._id.year, record._id.month - 1, record._id.day, record._id.hour, record._id.minute, record._id.quarter * 15, 0));
    if (currentRecordEventTime >= lastRecordedTimeForDriver)
      lastRecordedTimeForDriver = new Date(Date.UTC(record._id.year, record._id.month - 1, record._id.day, record._id.hour, record._id.minute, 59, 999));

    record['eventTime'] = currentRecordEventTime;
    record['eventId'] = record._id;
    delete record._id;
    record['accountId'] = record.accountIdArray[0];
    delete record.accountIdArray;

    record['vehicle'] = record.vehicleArray[0];
    delete record.vehicleArray;

    record['driver'] = record.driverArray[0];
    delete record.driverArray;

    record['driverVehicle'] = record.driverVehicleArray[0];
    delete record.driverVehicleArray;

    record.averageGPSLatitude = parseInt((record.averageGPSLatitude * 1000).toFixed(3)) / 1000;
    record.averageGPSLongitude = parseInt((record.averageGPSLongitude * 1000).toFixed(3)) / 1000;

    db.getCollection('vehicle_signature_records').insert(record);
    insertCounter += 1;
  });

  print(insertCounter + ' new records inserted. Last recorded time for device is: ' + lastRecordedTimeForDriver);

  // Save the end time of the device to the database
  if (driverIsNew) {  // which means this is a new device with no record
    db.book_keeping.update(
      {_id: 'driver_vehicles_processed_until'},
      {$push: {'lastEndTimes': {driver: driverName, endTime: lastRecordedTimeForDriver}}}
    );
  } else {
    var nowDate = new Date();
    db.book_keeping.update(
      {_id: 'driver_vehicles_processed_until', 'lastEndTimes.driver': driverName},
      {$set: {'lastEndTimes.$.endTime': lastRecordedTimeForDriver, 'lastEndTimes.$.driver': driverName}}  // lastRecordedTimeForDriver
    );
  }
});

sample_record = {
  "data" : {
    "load" : 94.11764705882354,
    "eng_fuel_rate" : "NODATA",
    "abs_throttle_pos_b" : 36.470588235294116,
    "dtc_ff" : "NODATA",
    "iccid" : "98107120720242459516",
    "rpm" : NumberInt(0),
    "throttle_pos" : 36.07843137254902,
    "intake_air_temp" : NumberInt(77),
    "speed" : 32.9397141081417,
    "altitude" : 21.3,
    "comm_throttle_ac" : 34.90196078431372,
    "secondary_air_status" : "NODATA",
    "location" : {
      "type" : "Point",
      "coordinates" : [
        -122.056615,
        37.387591667
      ]
    },
    "dtc_status" : [
      3.0,
      0.0,
      1.0,
      1.0,
      1.0,
      1.0,
      0.0,
      1.0,
      0.0,
      0.0,
      1.0,
      1.0,
      0.0
    ],
    "vehicle" : "gmc-denali-2015",
    "engine_time" : NumberInt(33),
    "username" : "grant",
    "eventtime" : "2017-07-18T19:13:08.183951",
    "abs_load" : 62.35294117647059,
    "rel_acc_pedal_pos" : "NODATA",
    "gear" : NumberInt(0),
    "rel_throttle_pos" : 0.0,
    "drv_demand_eng_torq" : "NODATA",
    "acc_pedal_pos_e" : 17.647058823529413,
    "acc_pedal_pos_d" : 36.470588235294116,
    "gps_speed" : 15.361,
    "short_term_fuel_trim_2" : NumberInt(-5),
    "o211" : NumberInt(2000),
    "o212" : NumberInt(31499),
    "short_term_fuel_trim_1" : NumberInt(-4),
    "maf" : 7.93656,
    "imei" : "353147040225957",
    "timing_advance" : 16.5,
    "climb" : 0.5,
    "fuel_pressure" : "5F",
    "temp" : NumberInt(201),
    "aux_input" : "NODATA",
    "o2_sensor_position_b" : "NODATA",
    "ambient_air_temp" : NumberInt(73),
    "manifold_pressure" : 599.8345284059569,
    "obd_standard" : "0B",
    "act_eng_torq" : "NODATA",
    "eng_ref_torq" : "NODATA",
    "fuel_status" : "0202",
    "long_term_fuel_trim_1" : NumberInt(-2),
    "heading" : 292.27,
    "long_term_fuel_trim_2" : NumberInt(-2),
    "eventTimestamp" : ISODate("2017-07-18T19:13:08.183+0000")
  },
  "driverVehicleId" : 11.0,
  "delta" : {
    "distance" : 0.04277374688521111,
    "interval" : 3503.0,
    "acceleration" : 0.7096820322880475,
    "angular_velocity" : -0.00014558949471880984,
    "incline" : 0.000371110476734228
  }
}
// End of program
