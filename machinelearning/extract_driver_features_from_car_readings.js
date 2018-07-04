// This script looks at the car_readings, and finds delta values between the subsequent records. It saves this data
// to the mldataset collection. This collection can then be used to determine driver signatures and rank drivers based on their behavior.
//
// mongo --nodb --quiet --eval "var host='localhost';" extract_driver_features_from_car_readings.js

/*
 What features are we going to extract? Here is a starter set:

 1) Over-speeding at low speed: When speed range is between 0-35miles/hour, how many times per mile did the driver go above the limit.
 2) Over-speeding at medium speed: When speed range is between 35-65miles/hour, how many times per mile did the driver go above the limit.
 3) Over-speeding at high speed: When speed range is above 65miles/hour, how many times per mile did the driver go above 75miles/hour.
 4) Max acceleration at low speed: When driver is starting from zero, what is the max acceleration value at start up.
 5) Max acceleration at medium speed: When driver is ramping up on the freeway (35 to 65 miles/hour), what is the max acceleration value.
 6) Max deceleration at low speed: When driver is driving between 0-35 miles/hour, what is the max deceleration value.
 7) Max deceleration at medium speed: When driver is driving between 35-65 miles/hour, what is the max deceleration value.
 8) Max deceleration at high speed: When driver is driving over 65miles/hour (probably on highway), what is the max deceleration value.
 9) Max throttle position when ramping up: What is the maximum throttle position when the driver is accelerating on the ramp when getting on a freeway.
 10) Hard left turns at low speed: When driver is driving between 0-35 miles/hour, how many hard left turn events per mile are we seeing.
 11) Hard left turns at medium speed: When driver is driving between 35-65 mile/hour, how many hard left turn events per mile are we seeing.
 12) Hard left turns at high speed: When driver is driving above 65 miles/hour, how many hard left turn events per mile are we seeing.
 13) Hard right turns at low speed: When driver is driving between 0-35 miles/hour, how many hard right turn events per mile are we seeing.
 14) Hard right turns at medium speed: When driver is driving between 35-65 mile/hour, how many hard right turn events per mile are we seeing.
 15) Hard right turns at high speed: When driver is driving above 65 miles/hour, how many hard right turn events per mile are we seeing.
 16) Max acceleration value at a slow U-turn: What is the maximum acceleration value when there is a sharp heading change and speed is below 10 miles/hour (how does the driver make a U-turn).
 17) Tailgating distance at various speeds: If a camera is present on the car, find out the tailgating distance, i.e. the average distance behind the car in front. This is based on speed, so the tailgating distance must be sampled for various speeds of the vehicle.
*/

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
var processedUntil = db.getCollection('book_keeping').findOne( { _id: "processed_until" } );
endTime = new Date();  // Set the end time to the current time
if (processedUntil != null) {
  startTime = processedUntil.lastEndTime;
} else {
  db.book_keeping.insert( { _id: "processed_until", lastEndTime: endTime } );
  startTime = new Date(endTime.getTime() - (365*86400000));  // Go back 365 days
}

// We need to do a string comparison on the car_readings table, so convert dates to a string
startTimeStr = startTime.toISOString();
endTimeStr = endTime.toISOString();

// Another book-keeping task is to read the driver-vehicle hash-table from the database
// Look up the book-keeping table to figure out the previous driver-vehicle codes (we have
// numbers representing the combination of drivers and vehicles).
var driverVehicles = db.getCollection('book_keeping').findOne( { _id: "driver_vehicles" } );
var drivers;
if (driverVehicles != null)
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
var allNewCarDrivers = db.getCollection('car_readings').aggregate([
  {
    "$match": {
      "timestamp" : { $gt: startTimeStr, $lte: endTimeStr  }
    }
  },
  {
    "$unwind": "$data"
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
  var allNewCarReadings = db.getCollection('car_readings').aggregate([
    {
      "$match": { // 1. Match all records that fall within the time range we have decided to use. Note that this is being
                  // done on a live database - which means that new data is coming in while we are trying to analyze it.
                  // Thus we have to pin both the starting time and the ending time. Pinning the endtime to the starting time
                  // of the application ensures that we will be accurately picking up only the NEW records when the program
                  // runs again the next time.
        "timestamp" : { $gt: startTimeStr, $lte: endTimeStr  }
      }
    },
    {
      $project: {  // We only need to consider a few fields for our analysis. This eliminates the summaries from our analysis.
        "timestamp": 1,
        "data" : 1,
        "account": 1,
        "_id": 0
      }
    },
    {
      $unwind: "$data"  // Flatten out all records into one gigantic array of records
    },
  	{
  	  $match: {
  	    "data.username": driverName  // Only consider the records for this specific driver, ignoring all others.
  	  }
  	},
    {
      $sort: {
        "data.eventtime": 1  // Finally sort the data based on eventtime in ascending order
      }
    }
  ]);

  var lastRecord = null; // We create a variable to remember what was the last record processed

  var numProcessedRecords = 0;
  allNewCarReadings.forEach(function(record) {
    // Here we are reading a raw record from the car_readings collection, and then enhancing it with a few more
    // variables. These are the (1) id of the driver-vehicle combination and (2) the delta values between current and previous record
    numProcessedRecords += 1;  // This is just for printing number of processed records when the program is running
    var lastTime;  // This is the timestamp of the last record
    if (lastRecord !== null) {
      lastTime = lastRecord.data.eventtime;
    } else {
      lastTime = "";
    }
    var eventTime = record.data.eventtime;
    record.data.eventTimestamp = new Date(record.data.eventtime+'Z');  // Creating a real timestamp from an ISO string (without the trailing 'Z')
    // print('Eventtime = ' + eventTime);
    if (eventTime !== lastTime) {  // this must be a new record
      var driverVehicle = record.data.vehicle + "_" + record.data.username;
      if (drivers.hasOwnProperty(driverVehicle))
        record.driverVehicleId = drivers[driverVehicle];
      else {
        drivers[driverVehicle] = maxDriverVehicleId;
        record.driverVehicleId = maxDriverVehicleId;
        maxDriverVehicleId += 1;
      }

      record.delta = {};  // delta stores the difference between the current record and the previous record
      if (lastRecord !== null) {
        var timeDifference = record.data.eventTimestamp.getTime() - lastRecord.data.eventTimestamp.getTime();  // in milliseconds
        record.delta["distance"] = earth_distance_havesine(
          record.data.location.coordinates[1],
          record.data.location.coordinates[0],
          lastRecord.data.location.coordinates[1],
          lastRecord.data.location.coordinates[0],
          "K");
        if (timeDifference < 60000) {
          // if time difference is less than 60 seconds, only then can we consider it as part of the same session
          // print(JSON.stringify(lastRecord.data));
          record.delta["interval"] = timeDifference;
          record.delta["acceleration"] = 1000 * (record.data.speed - lastRecord.data.speed) / timeDifference;
          record.delta["angular_velocity"] = (record.data.heading - lastRecord.data.heading) / timeDifference;
          record.delta["incline"] = (record.data.altitude - lastRecord.data.altitude) / timeDifference;
        } else {
          // otherwise this is a new session. So we still store the records, but the delta calculation is all set to zero.
          record.delta["interval"] = timeDifference;
          record.delta["acceleration"] = 0.0;
          record.delta["angular_velocity"] = 0.0;
          record.delta["incline"] = 0.0;
        }
        db.getCollection('mldataset').insert(record);
      }
    }
    if (numProcessedRecords % 100 === 0)
      print("Processed " + numProcessedRecords + " records");
    lastRecord = record;
  });
});

db.book_keeping.update(
  { _id: "driver_vehicles"},
  { $set: { drivers: drivers } },
  { upsert: true }
);

// Save the end time to the database
db.book_keeping.update(
  { _id: "processed_until" },
  { $set: { lastEndTime: endTime } },
  { upsert: true }
);

// End of program

// Below is an example record the car_readings collection, just for reference.
sample_record = {
  "load" : 26.666666666666668,
  "iccid" : "98107120720242459516",
  "rpm" : 1162,
  "timing_advance" : 10.0,
  "speed" : 11.808576755748913,
  "altitude" : -9.7,
  "location" : {
    "type" : "Point",
    "coordinates" : [
      -121.967758333,
      37.38948
    ]
  },
  "vehicle" : "subaru-outback-2015",
  "username" : "jins",
  "eventtime" : "2017-06-07T21:20:18.996979",
  "gear" : 2,
  "intake_air_temp" : 95,
  "gps_speed" : 5.314,
  "short_term_fuel_trim_2" : "NODATA",
  "o212" : NumberInt(32400),
  "short_term_fuel_trim_1" : 1,
  "maf" : 0.5251357200000001,
  "imei" : "353147040225957",
  "throttle_pos" : 14.901960784313726,
  "climb" : 0.3,
  "temp" : 159,
  "fuel_status" : "0200",
  "long_term_fuel_trim_1" : -4,
  "heading" : 244.06,
  "long_term_fuel_trim_2" : "NODATA"
}
