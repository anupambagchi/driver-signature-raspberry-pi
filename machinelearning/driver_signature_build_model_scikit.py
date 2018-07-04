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
#  python driver_signature_build_model_scikit.py localhost driver

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from bson import json_util
from datetime import datetime, timedelta
from itertools import cycle
from pandas.io.json import json_normalize
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support as prf
import operator

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

        # Then choose only a small subset of the data, frac=1 means choose everything
        df = df.sample(frac=1, replace=True)

        df.fillna(value=0, inplace=True)

        train_df, test_df, validate_df = np.split(df, [int(.8*len(df)), int(.9*len(df))])

        df[feature_name] = df[feature_name].astype('category')

        #print('Sample dataframe given below:')
        #print(df.head())
        #print(df.dtypes)

        y_train = train_df[class_variables]
        X_train = train_df.reindex(columns=features)
        X_train.replace('NODATA', 0, regex=False, inplace=True)
        X_train.fillna(value=0, inplace=True)

        y_test = test_df[class_variables]
        X_test = test_df.reindex(columns=features)
        X_test.replace('NODATA', 0, regex=False, inplace=True)
        X_test.fillna(value=0, inplace=True)

        y_validate = validate_df[class_variables]
        X_validate = validate_df.reindex(columns=features)
        X_test.replace('NODATA', 0, regex=False, inplace=True)
        X_validate.fillna(value=0, inplace=True)

        print('Sample X train data given below:')
        print('Shape: ' + str(X_train.shape))
        print(X_train.head())
        print('Sample Y train data given below:')
        print('Shape: ' + str(y_train.shape))
        print(y_train.head())

        print('Sample X test data given below:')
        print('Shape: ' + str(X_test.shape))
        print(X_test.head())
        print('Sample Y test data given below:')
        print('Shape: ' + str(y_test.shape))
        print(y_test.head())

        print('Sample X validate data given below:')
        print('Shape: ' + str(X_validate.shape))
        print(X_validate.head())
        print('Sample Y validate data given below:')
        print('Shape: ' + str(y_validate.shape))
        print(y_validate.head())

        X_train.to_csv('X_train.csv')
        y_train.to_csv('y_train.csv')

        X_test.to_csv('X_test.csv')
        y_test.to_csv('y_test.csv')

        X_validate.to_csv('X_validate.csv')
        y_validate.to_csv('y_validate.csv')

        print('Source data has been saved as CSV files in current folder.')
    else:
        X_train = pd.read_csv(filepath_or_buffer='X_train.csv', index_col=0)
        y_train = pd.read_csv(filepath_or_buffer='y_train.csv', index_col=0)

        X_test = pd.read_csv(filepath_or_buffer='X_test.csv', index_col=0)
        y_test = pd.read_csv(filepath_or_buffer='y_test.csv', index_col=0)

        X_validate = pd.read_csv(filepath_or_buffer='X_validate.csv', index_col=0)
        y_validate = pd.read_csv(filepath_or_buffer='y_validate.csv', index_col=0)


    model_file = 'driver_signature_random_forest_model_'+feature_name+'.pkl'
    if read_and_proceed:
        print('Reading learned model from ' + model_file)
        dt = joblib.load(model_file)
    else:
        print ('Building RandomForest Classifier ...')

        dt = RandomForestClassifier(n_estimators=20, min_samples_leaf=1, max_depth=20, min_samples_split=2, random_state=0)
        dt.fit(X_train, y_train.values.ravel())

        joblib.dump(dt, model_file)
        print('...done. Your Random Forest classifier has been saved in file: ' + model_file)

    print('Completed generating or loading Learning Model')
    # Test the model to find out how accurate it is
    y_pred = dt.predict(X_test)
    y_test_as_matrix = y_test.as_matrix()
    print('Completed generating predicted set')

    print ('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))

    #print(type(y_pred), len(y_pred))

    #crossValScore = cross_val_score(dt, X_validate, y_validate)

    model_score = dt.score(X_test, y_test_as_matrix)

    correct = 0
    incorrect = 0
    for k in range(0, len(y_test_as_matrix)):
        if y_test_as_matrix[k] == y_pred[k]:
            correct += 1
        else:
            incorrect += 1

    model_score = dt.score(X_test, y_test_as_matrix)

    correct = 0
    incorrect = 0
    for k in range(0, len(y_test_as_matrix)):
        if y_test_as_matrix[k] == y_pred[k]:
            correct += 1
        else:
            incorrect += 1

    print('Features: ' + str(features))
    #print('Feature importances: ', dt.feature_importances_)
    print('Feature importances:')

    # Create a dictionary from the feature_importances_ index -> importance_value
    importance_indices = {}
    for z in range(0, len(dt.feature_importances_)):
        importance_indices[z] = dt.feature_importances_[z]

    sorted_importance_indices = sorted(importance_indices.items(), key=operator.itemgetter(1), reverse=True)

    for k1 in sorted_importance_indices:
        print(features[int(k1[0])] + ' -> ' + str(k1[1]))

    print('Total values checked: %d' % (correct + incorrect))
    print('Correct values: %d' % correct)
    print('Incorrect values: %d' % incorrect)
    print('Model Score: %f' % model_score)
    #print('Cross validation score: ', crossValScore)
    print('Percentage correct: %5.1f%%' % ((100 * correct) / (correct + incorrect)))

    print("F1 Score with macro averaging:" + str(f1_score(y_test, y_pred, average='macro')))
    print("F1 Score with micro averaging:" + str(f1_score(y_test, y_pred, average='micro')))
    print("F1 Score with weighted averaging:" + str(f1_score(y_test, y_pred, average='weighted')))

    print ('Precision, Recall and FScore')
    precision, recall, fscore, _ = prf(y_test, y_pred, pos_label=1, average='micro')
    print('Precision: ' + str(precision))
    print('Recall:' + str(recall))
    print('FScore:' + str(fscore))

    exit()

    '''
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_pred[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_pred, average="micro")

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw = 2

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[0], precision[0], lw=lw, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()

    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    #visualize_tree(dt, list(features))

    exit()
    '''

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('You need to provide the database host and one of {\'all\', \'driver\', \'vehicle\'} as a parameter')
        print('e.g. ' + sys.argv[0] + ' localhost all')
        exit(-1)
    main(sys.argv[1:])
