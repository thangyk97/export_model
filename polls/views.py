from django.shortcuts import render
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import numpy as np
from .utils import decode_netout, draw_boxes
import cv2
# Create your views here.
from django.http import HttpResponse

LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
CLASS = len(LABELS)

def index(request):
# def index():
    img =  cv2.imread('/home/vtc/git/export_model/polls/dog.jpg')
    input_image = cv2.resize(img, (416, 416))

    input_image = input_image / 255.
    input_image = input_image[:, :, ::-1]
    input_image = np.float32(np.expand_dims(input_image, 0))


    # output = do_inference('localhost:9000', '/tmp', 1, 1000)
    host, port = 'localhost', '9000'
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'yolo'
    request.model_spec.signature_name = 'predict'
    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(input_image))

    result = stub.Predict(request, 10.0)
    a = result.outputs['boxs'].float_val
    a = np.array(a)
    a = np.reshape(a, [13, 13, 5, 85])
    boxs = decode_netout(a,
                          obj_threshold=0.3,
                          nms_threshold=0.3,
                          anchors=ANCHORS,
                          nb_class=CLASS)
    list = draw_boxes(img, boxs, labels=LABELS)
    print (list)
    return HttpResponse(list)