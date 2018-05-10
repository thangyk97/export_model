from django.shortcuts import render
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import numpy as np
from .utils import decode_netout, get_info, get_distance, get_top_boxes
import cv2
from PIL import Image
from io import BytesIO
# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import TemplateView
from django.shortcuts import render_to_response
from django.views.decorators.csrf import csrf_exempt


# LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
LABELS = ['người', 'xe đạp', 'xe hơi', 'xe máy', 'máy bay', 'xe buýt', 'tàu', 'xe tải', 'thuyền', 'đèn giao thông', 'vòi cứu hỏa', 'dừng ký ',' đồng hồ đỗ xe ',' băng ghế ',' chim ',' mèo ',' chó ',' ngựa ',' cừu ',' bò ',' voi ',' gấu ',' ngựa vằn ',' hươu cao cổ ' , 'ba lô', 'ô', 'túi xách', 'cà vạt', 'vali', 'frisbee', 'ván trượt', 'ván trượt tuyết', 'bóng thể thao', 'diều', 'gậy bóng chày', 'găng tay bóng chày ',' ván trượt ',' ván lướt sóng ',' vợt tennis ',' chai ',' ly rượu ',' chén ',' ngã ba ',' dao ',' muỗng ',' bát ',' chuối ',' táo ',' sandwich ',' cam ',' bông cải xanh ',' cà rốt ',' chó nóng ',' pizza ',' bánh rán ',' bánh ',' ghế ',' ghế ',' chậu cây ',' giường ',' bàn ăn ',' nhà vệ sinh ',' tv ',' máy tính xách tay ',' chuột ',' từ xa ',' bàn phím ',' điện thoại di động ',' lò vi sóng ',' lò ',' lò nướng bánh ',' bồn rửa ',' tủ lạnh ',' sách ',' đồng hồ ',' bình ',' kéo ',' gấu bông ',' máy sấy tóc ',' bàn chải đánh răng ']
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
CLASS = len(LABELS)

class UploadImg(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'upload_img.html', context=None)

class upload_2_img(TemplateView):
    def get(self, request, ** kwargs):
        return render(request, 'upload_img_measure_distance.html', context=None)
        
@csrf_exempt
def index(request):
    s = ""
    if request.method == 'POST':
        image = request.FILES['image']
        imagefile  = BytesIO(image.read())
        imageImage = Image.open(imagefile)
        a, img = predict(imageImage)
        
        boxs = decode_netout(a,
                            obj_threshold=0.3,
                            nms_threshold=0.3,
                            anchors=ANCHORS,
                            nb_class=CLASS)
                            
        s = get_info(img, boxs, labels=LABELS)

    # return render_to_response('show_result.html', {'s': s})
    return HttpResponse(s)
@csrf_exempt
def measure_distance(request):
    s = ""
    if request.method == 'POST':
        image1 = request.FILES['image1']
        image2 = request.FILES['image2']

        imagefile1  = BytesIO(image1.read())
        imagefile2  = BytesIO(image2.read())

        imageImage1 = Image.open(imagefile1)
        imageImage2 = Image.open(imagefile2)
        a1, img1 = predict(imageImage1)
        a2, img2 = predict(imageImage2)
        
        boxs1 = decode_netout(a1,
                            obj_threshold=0.3,
                            nms_threshold=0.3,
                            anchors=ANCHORS,
                            nb_class=CLASS)
        
        boxs2 = decode_netout(a2,
                            obj_threshold=0.3,
                            nms_threshold=0.3,
                            anchors=ANCHORS,
                            nb_class=CLASS)

        boxs1 = get_top_boxes(boxs1, 1)
        boxs2 = get_top_boxes(boxs2, 1)

        s = get_distance(boxs1, boxs2, 0.5, LABELS)

    return HttpResponse(s)




def predict(image):
    """
    
    
    """
    img = np.array(image)

    input_image = cv2.resize(img, (416, 416))

    input_image = input_image / 255.
    input_image = input_image[:, :, ::-1]
    input_image = np.float32(np.expand_dims(input_image, 0))

    host, port = 'localhost', '9000'
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'yolo'
    request.model_spec.signature_name = 'predict'
    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(input_image))

    result = stub.Predict(request, 10.0)
    a = result.outputs['boxes'].float_val
    a = np.array(a)
    a = np.reshape(a, [13, 13, 5, 85])
    return a, img