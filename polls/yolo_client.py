from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

host, port = 'localhost', '9000'
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'yolo'
request.model_spec.signature_name = 'predict'
request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(img))

result = stub.Predict(request, 10.0)
