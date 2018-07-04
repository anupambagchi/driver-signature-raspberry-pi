#!/usr/bin/env python
import sys
sys.path.insert(0,'../src')

from flask import Flask, Blueprint
from flask import request, jsonify
from flask_restful import Api
from kafka import KafkaProducer
import zlib
import base64
import pymongo
import json
from Crypto.PublicKey import RSA
from AESCipher import AESCipher
from PKS1_OAEPCipher import PKS1_OAEPCipher

# Note you need do 'pip install flask_restful', 'pip install kafka-python' and 'pip install sklearn'
client = pymongo.MongoClient('localhost', 27017)
mongo_database = client.obd2
mongo_collection = mongo_database.car_readings

# Handle the decryption keys once when program starts
decryptionKeyHandle = open('decryption.key', 'r')
decryptionKey = RSA.importKey(decryptionKeyHandle.read())
decryptionKeyHandle.close()

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

app = Flask(__name__)
api_bp = Blueprint('api', __name__)
api = Api(api_bp)

app.register_blueprint(api_bp)

@app.route('/')
def index():
    return 'OBD-catcher alive !'

# http://127.0.0.1:5000/obd2/v1/17530/upload
@app.route('/obd2/api/v1/<account>/upload', methods=['POST'])
def save_record(account):
    print(request.content_type)
    if not request.json or not 'size' in request.json:
        raise InvalidUsage('Invalid usage of this web-service detected', status_code=400)

    size = int(request.json['size'])
    decoded_compressed_record = request.json.get('data', "")
    symmetricKeyEncrypted = request.json.get('key', "")

    compressed_record = base64.b64decode(decoded_compressed_record)
    encrypted_json_record_str = zlib.decompress(compressed_record)

    pks1OAEPForDecryption = PKS1_OAEPCipher()
    pks1OAEPForDecryption.readDecryptionKey('decryption.key')
    symmetricKeyDecrypted = pks1OAEPForDecryption.decrypt(base64.b64decode(symmetricKeyEncrypted))

    aesCipherForDecryption = AESCipher()
    aesCipherForDecryption.setKey(symmetricKeyDecrypted)

    json_record_str = aesCipherForDecryption.decrypt(encrypted_json_record_str)

    record_as_dict = json.loads(json_record_str)

    # Add the account ID to the reading here
    record_as_dict["account"] = account

    #print record_as_dict
    post_id = mongo_collection.insert_one(record_as_dict).inserted_id
    print('Saved as Id: %s' % post_id)

    producer = KafkaProducer(bootstrap_servers=['your.kafka.server.com:9092'],
                             value_serializer=lambda m: json.dumps(m).encode('ascii'),
                             retries=5)
    # send the individual records to the Kafka queue for stream processing
    raw_readings = record_as_dict["data"]
    counter = 0
    for raw_reading in raw_readings:
        raw_reading["id"] = str(post_id) + str(counter)
        raw_reading["account"] = account
        producer.send("car_readings", raw_reading)
        counter += 1

    producer.flush()
    # send the summary to the Kafka queue in case there is some stream processing required for that as well
    raw_summary = record_as_dict["summary"]
    raw_summary["id"] = str(post_id)
    raw_summary["account"] = account
    raw_summary["eventTime"] = record_as_dict["timestamp"]
    producer.send("car_summaries", raw_summary)

    producer.flush()
    return jsonify({'title': str(size) + ' bytes received'}), 201

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
