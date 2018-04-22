from django.shortcuts import render
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import numpy as np

# Create your views here.
from django.http import HttpResponse


def index(request):
# def index():
    img = np.ones([1,416,416, 3], dtype=np.float32)

    # output = do_inference('localhost:9000', '/tmp', 1, 1000)
    host, port = 'localhost', '9000'
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'yolo'
    request.model_spec.signature_name = 'predict'
    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(img))

    result = stub.Predict(request, 10.0)
    a = result.outputs['boxs'].float_val
    a = np.array(a)
    a = np.reshape(a, [13, 13, 5, 85])
    print (a.shape)
    return HttpResponse(a)