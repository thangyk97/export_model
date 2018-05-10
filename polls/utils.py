import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy

class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def normalize(image):
    image = image / 255.
    
    return image

def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2
    
    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2
    
    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    
    intersect = intersect_w * intersect_h
    
    union = box1.w * box1.h + box2.w * box2.h - intersect
    
    return float(intersect) / union
    
def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3  

def get_top_boxes(boxes, top_size=1):
    """
    
    """
    top_size = min([len(boxes), top_size])

    squares  = [compute_s(box) for box in boxes]
    indexes = index_top_size(squares, top_size) # List
    result = [boxes[i] for i in indexes]

    return result

def get_distance(boxes1, boxes2, delta, labels):
    """
    
    """
    s = ""
    for i in range(min([len(boxes1), len(boxes2)])):
        d = distance(boxes1[i], boxes2[i], delta)
        s += labels[boxes1[i].get_label()] + " " + str(format(d, '.2f')) + " mét, "

    return s

def compute_s(box):
    return box.w * box.h

def index_top_size(squares, top_size):
    i = 0
    max = 0
    for k in range(len(squares)):
        if squares[k] > max:
            max = squares[k]
            i   = k

    return [i]

def distance(box1, box2, delta):
    temp = min([box1.h, box2.h]) / max([box1.h, box2.h])
    if 1 - temp != 0:
        x1 = delta / (1 - temp)
    else:
        x1 = 0
    temp = min([box1.w, box2.w]) / max([box1.w, box2.w])
    if 1 - temp != 0:
        x2 = delta / (1 - temp)
    else:
        x2 = 0

    return (x1 + x2) / 2


def get_info(image, boxes, labels):
    """
    detect objectes in image
    count same objects
    compute position of object
    """
    result = ""
    list_left = []
    list_right = []
    list_midder = []
    objects_left = np.zeros(len(labels), np.int64)
    objects_right = np.zeros(len(labels), np.int64)
    objects_midder = np.zeros(len(labels), np.int64)

    for box in boxes:
        # xmin  = int((box.x - box.w/2) * image.shape[1])
        # xmax  = int((box.x + box.w/2) * image.shape[1])
        # ymin  = int((box.y - box.h/2) * image.shape[0])
        # ymax  = int((box.y + box.h/2) * image.shape[0])
        
        
        left = (1 / 3)
        right = left * 2

        if box.x < left:
            objects_left[box.label] += 1
        elif box.x > right:
            objects_right[box.label] += 1
        else:
            objects_midder[box.label] += 1

        # list.append(labels[box.get_label()] + ' ' + str(int(round(box.get_score() * 100))))

    for i in range(len(labels)):
        if objects_left[i] > 0:
            list_left.append(str(objects_left[i]) + ' ' + labels[i])
        if objects_right[i] > 0:
            list_right.append(str(objects_right[i]) + ' ' + labels[i])
        if objects_midder[i] > 0:
            list_midder.append(str(objects_midder[i]) + ' ' + labels[i])  
    if len(list_midder) > 0 : result += "phía trước có "
    for s in list_midder:
        result += s + " "
    if len(list_left) > 0 : result += ", bên trái có "
    for s in list_left:
        result += s + " "
    if len(list_right) > 0 : result += ", bên phải có "
    for s in list_right:
        result += s + " "

    return result
        
def decode_netout(netout, obj_threshold, nms_threshold, anchors, nb_class):

    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold
    
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if classes.any():
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x, y, w, h, confidence, classes)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)

