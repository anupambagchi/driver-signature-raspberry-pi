# Compare Algorithms

#!/usr/bin/env python3

__author__ = 'Anupam Bagchi'

'''
This program builds a driver signature model per account from readings aggregated over 15 minute intervals.
It creates a random-forest model by reading the aggregated values directly from the database. We assume that the 
aggregation of the device records is happening periodically. See extract_driver_features_from_car_readings.js 
and extract_features_from_mldataset.js to know how the aggregation is being done. The aggregated data for each vehicle
(per 15 minutes) is stored in the collection 'vehicle_signature_records'.
The created learning model is stored in the current directory where this python program is running.
'''

# Typical way to call this is:
# python compare-learning-models.py localhost driver

import json
import numpy as np
import pandas as pd
import sys
from bson import json_util
from datetime import datetime, timedelta
from itertools import cycle
from pandas.io.json import json_normalize
from pymongo import MongoClient

from sklearn.externals import joblib
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support as prf
import operator

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

# np.random.seed(1671)  # for reproducibility

__author__ = 'Anupam Bagchi'

def _connect_mongo(host, port, username, password, db):
    """ A utility for making a connection to MongoDB """

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)

    return conn[db]

def read_mongo(db, collection, query={}, projection='', limit=1000, host='localhost', port=27017, username=None, password=None, no_id=False):
    """ Read from Mongo and Store into DataFrame """

    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query, projection).limit(limit)

    # Expand the cursor and construct the DataFrame
    datalist = list(cursor)
    #print(datalist)

    sanitized = json.loads(json_util.dumps(datalist))
    normalized = json_normalize(sanitized)
    df = pd.DataFrame(normalized)

    #df = pd.DataFrame(datalist)

    # Delete the _id
    #if no_id:
    #    del df['_id']

    return df

def main(argv):
    DATABASE_HOST = argv[0]
    CHOSEN_FEATURE_SET = argv[1]

    readFromDatabase = True
    read_and_proceed = False

    # Make a MongoDB query and find out all accounts active between the specified time range
    endTime = datetime.now()
    hours24 = timedelta(days=200)
    startTime = endTime - hours24

    all_features = [
        "averageGPSLatitude",
        "averageGPSLongitude",
        "averageLoad",
        "minLoad",
        "maxLoad",
        "averageThrottlePosB",
        "minThrottlePosB",
        "maxThrottlePosB",
        "averageRpm",
        "minRpm",
        "maxRpm",
        "averageThrottlePos",
        "minThrottlePos",
        "maxThrottlePos",
        "averageIntakeAirTemp",
        "minIntakeAirTemp",
        "maxIntakeAirTemp",
        "averageSpeed",
        "minSpeed",
        "maxSpeed",
        "averageAltitude",
        "minAltitude",
        "maxAltitude",
        "averageCommThrottleAc",
        "minCommThrottleAc",
        "maxCommThrottleAc",
        "averageEngineTime",
        "minEngineTime",
        "maxEngineTime",
        "averageAbsLoad",
        "minAbsLoad",
        "maxAbsLoad",
        "averageGear",
        "minGear",
        "maxGear",
        "averageRelThrottlePos",
        "minRelThrottlePos",
        "maxRelThrottlePos",
        "averageAccPedalPosE",
        "minAccPedalPosE",
        "maxAccPedalPosE",
        "averageAccPedalPosD",
        "minAccPedalPosD",
        "maxAccPedalPosD",
        "averageGpsSpeed",
        "minGpsSpeed",
        "maxGpsSpeed",
        "averageShortTermFuelTrim2",
        "minShortTermFuelTrim2",
        "maxShortTermFuelTrim2",
        "averageO211",
        "minO211",
        "maxO211",
        "averageO212",
        "minO212",
        "maxO212",
        "averageShortTermFuelTrim1",
        "minShortTermFuelTrim1",
        "maxShortTermFuelTrim1",
        "averageMaf",
        "minMaf",
        "maxMaf",
        "averageTimingAdvance",
        "minTimingAdvance",
        "maxTimingAdvance",
        "averageClimb",
        "minClimb",
        "maxClimb",
        #"averageFuelPressure",
        #"minFuelPressure",
        #"maxFuelPressure",
        "averageTemp",
        "minTemp",
        "maxTemp",
        "averageAmbientAirTemp",
        "minAmbientAirTemp",
        "maxAmbientAirTemp",
        "averageManifoldPressure",
        "minManifoldPressure",
        "maxManifoldPressure",
        "averageLongTermFuelTrim1",
        "minLongTermFuelTrim1",
        "maxLongTermFuelTrim1",
        "averageLongTermFuelTrim2",
        "minLongTermFuelTrim2",
        "maxLongTermFuelTrim2",
        "averageGPSAcceleration",
        "minGPSAcceleration",
        "maxGPSAcceleration",
        "averageHeadingChange",
        "minHeadingChange",
        "maxHeadingChange",
        #"averageIncline",   # Has zero importance for the model
        #"minIncline",
        #"maxIncline",
        "averageAcceleration",
        "minAcceleration",
        "maxAcceleration"
    ]

    # These are the features that define a vehicle's characteristics
    vehicle_features = [
        #"averageGPSLatitude",
        #"averageGPSLongitude",
        "averageLoad",
        "minLoad",
        "maxLoad",
        #"averageThrottlePosB",
        #"minThrottlePosB",
        #"maxThrottlePosB",
        "averageRpm",
        "minRpm",
        "maxRpm",
        #"averageThrottlePos",
        #"minThrottlePos",
        #"maxThrottlePos",
        #"averageIntakeAirTemp",
        #"minIntakeAirTemp",
        #"maxIntakeAirTemp",
        #"averageSpeed",
        #"minSpeed",
        #"maxSpeed",
        #"averageAltitude",
        #"minAltitude",
        #"maxAltitude",
        #"averageCommThrottleAc",
        #"minCommThrottleAc",
        #"maxCommThrottleAc",
        "averageEngineTime",
        "minEngineTime",
        "maxEngineTime",
        "averageAbsLoad",
        "minAbsLoad",
        "maxAbsLoad",
        #"averageGear",
        #"minGear",
        #"maxGear",
        #"averageRelThrottlePos",
        #"minRelThrottlePos",
        #"maxRelThrottlePos",
        "averageAccPedalPosE",
        "minAccPedalPosE",
        "maxAccPedalPosE",
        "averageAccPedalPosD",
        "minAccPedalPosD",
        "maxAccPedalPosD",
        #"averageGpsSpeed",
        #"minGpsSpeed",
        #"maxGpsSpeed",
        "averageShortTermFuelTrim2",
        "minShortTermFuelTrim2",
        "maxShortTermFuelTrim2",
        "averageO211",
        "minO211",
        "maxO211",
        "averageO212",
        "minO212",
        "maxO212",
        "averageShortTermFuelTrim1",
        "minShortTermFuelTrim1",
        "maxShortTermFuelTrim1",
        "averageMaf",
        "minMaf",
        "maxMaf",
        "averageTimingAdvance",
        "minTimingAdvance",
        "maxTimingAdvance",
        #"averageClimb",
        #"minClimb",
        #"maxClimb",
        #"averageFuelPressure",
        #"minFuelPressure",
        #"maxFuelPressure",
        "averageTemp",
        "minTemp",
        "maxTemp",
        #"averageAmbientAirTemp",
        #"minAmbientAirTemp",
        #"maxAmbientAirTemp",
        "averageManifoldPressure",
        "minManifoldPressure",
        "maxManifoldPressure",
        "averageLongTermFuelTrim1",
        "minLongTermFuelTrim1",
        "maxLongTermFuelTrim1",
        "averageLongTermFuelTrim2",
        "minLongTermFuelTrim2",
        "maxLongTermFuelTrim2"
        #"averageGPSAcceleration",
        #"minGPSAcceleration",
        #"maxGPSAcceleration",
        #"averageHeadingChange",
        #"minHeadingChange",
        #"maxHeadingChange",
        #"averageIncline",
        #"minIncline",
        #"maxIncline",
        #"averageAcceleration",
        #"minAcceleration",
        #"maxAcceleration"
    ]

    # These are the features that define a driver's behavior
    driver_features = [
        "averageGPSLatitude",
        "averageGPSLongitude",
        #"averageLoad",
        #"minLoad",
        #"maxLoad",
        "averageThrottlePosB",
        "minThrottlePosB",
        "maxThrottlePosB",
        #"averageRpm",
        #"minRpm",
        #"maxRpm",
        "averageThrottlePos",
        "minThrottlePos",
        "maxThrottlePos",
        "averageIntakeAirTemp",
        "minIntakeAirTemp",
        "maxIntakeAirTemp",
        "averageSpeed",
        "minSpeed",
        "maxSpeed",
        "averageAltitude",
        "minAltitude",
        "maxAltitude",
        "averageCommThrottleAc",
        "minCommThrottleAc",
        "maxCommThrottleAc",
        #"averageEngineTime",
        #"minEngineTime",
        #"maxEngineTime",
        #"averageAbsLoad",
        #"minAbsLoad",
        #"maxAbsLoad",
        "averageGear",
        "minGear",
        "maxGear",
        "averageRelThrottlePos",
        "minRelThrottlePos",
        "maxRelThrottlePos",
        #"averageAccPedalPosE",
        #"minAccPedalPosE",
        #"maxAccPedalPosE",
        #"averageAccPedalPosD",
        #"minAccPedalPosD",
        #"maxAccPedalPosD",
        "averageGpsSpeed",
        "minGpsSpeed",
        "maxGpsSpeed",
        #"averageShortTermFuelTrim2",
        #"minShortTermFuelTrim2",
        #"maxShortTermFuelTrim2",
        #"averageO211",
        #"minO211",
        #"maxO211",
        #"averageO212",
        #"minO212",
        #"maxO212",
        #"averageShortTermFuelTrim1",
        #"minShortTermFuelTrim1",
        #"maxShortTermFuelTrim1",
        #"averageMaf",
        #"minMaf",
        #"maxMaf",
        #"averageTimingAdvance",
        #"minTimingAdvance",
        #"maxTimingAdvance",
        "averageClimb",
        "minClimb",
        "maxClimb",
        #"averageFuelPressure",
        #"minFuelPressure",
        #"maxFuelPressure",
        #"averageTemp",
        #"minTemp",
        #"maxTemp",
        "averageAmbientAirTemp",
        "minAmbientAirTemp",
        "maxAmbientAirTemp",
        #"averageManifoldPressure",
        #"minManifoldPressure",
        #"maxManifoldPressure",
        #"averageLongTermFuelTrim1",
        #"minLongTermFuelTrim1",
        #"maxLongTermFuelTrim1",
        #"averageLongTermFuelTrim2",
        #"minLongTermFuelTrim2",
        #"maxLongTermFuelTrim2",
        "averageGPSAcceleration",
        "minGPSAcceleration",
        "maxGPSAcceleration",
        "averageHeadingChange",
        "minHeadingChange",
        "maxHeadingChange",
        #"averageIncline",
        #"minIncline",
        #"maxIncline",
        "averageAcceleration",
        "minAcceleration",
        "maxAcceleration"
    ]

    feature_name = 'driverVehicleId'
    if (CHOSEN_FEATURE_SET == 'vehicle'):
        features = vehicle_features
        feature_name = 'vehicle'
        class_variables = ['vehicle']  # Declare the vehicle as a class variable
    elif (CHOSEN_FEATURE_SET == 'driver'):
        features = driver_features
        feature_name = 'driver'
        class_variables = ['driver']  # Declare the driver as a class variable
    else:
        features = all_features
        feature_name = 'driverVehicleId'
        class_variables = ['driverVehicleId']  # Declare the driver-vehicle combo as a class variable

    if readFromDatabase:
        if CHOSEN_FEATURE_SET == 'driver':  # Choose the records only for one vehicle which has multiple drivers
            df = read_mongo('obd2', 'vehicle_signature_records', {"vehicle": {"$regex" : ".*gmc-denali.*"}, "eventTime": {"$gte": startTime, "$lte": endTime} }, {"_id": 0}, 1000000, DATABASE_HOST, 27017, None, None, True )
        else:
            df = read_mongo('obd2', 'vehicle_signature_records', {"eventTime": {"$gte": startTime, "$lte": endTime} }, {"_id": 0}, 1000000, DATABASE_HOST, 27017, None, None, True )

        print(df.describe())
        #print(df)

        # First randomize the entire dataset
        df = df.sample(frac=1).reset_index(drop=True)
        df.fillna(value=0, inplace=True)

        train_df, test_df, validate_df = np.split(df, [int(1.0*len(df)), int(1.0*len(df))])

        df[feature_name] = df[feature_name].astype('category')

        #print('Sample dataframe given below:')
        #print(df.head())
        #print(df.dtypes)

        y_train = train_df[class_variables]
        X_train = train_df.reindex(columns=features)
        X_train.replace('NODATA', 0, regex=False, inplace=True)
        X_train.fillna(value=0, inplace=True)

        print('Sample X train data given below:')
        print('Shape: ' + str(X_train.shape))
        print(X_train.head())
        print('Sample Y train data given below:')
        print('Shape: ' + str(y_train.shape))
        print(y_train.head())
    else:
        X_train = pd.read_csv(filepath_or_buffer='X_train.csv', index_col=0)
        y_train = pd.read_csv(filepath_or_buffer='y_train.csv', index_col=0)

    seed = 7
    # Prepare models
    models = []
    models.append(('Log Reg', LogisticRegression()))
    models.append(('Lin Disc Anal', LinearDiscriminantAnalysis()))
    models.append(('K-Nearest Neigh', KNeighborsClassifier()))
    models.append(('Class & Reg Tech', DecisionTreeClassifier()))
    models.append(('Gaussian NB', GaussianNB()))
    models.append(('Sup Vec Mac', SVC()))
    models.append(('Rand Forest', RandomForestClassifier()))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('You need to provide the database host and one of {\'all\', \'driver\', \'vehicle\'} as a parameter')
        print('e.g. ' + sys.argv[0] + ' localhost all')
        exit(-1)
    main(sys.argv[1:])


