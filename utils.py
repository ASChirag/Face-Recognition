import cv2
import numpy as np

def load_detection_model():
    detector_model = "res10_300x300_ssd_iter_140000.caffemodel"
    detector_config = "deploy.prototxt"
    return cv2.dnn.readNetFromCaffe(detector_config, detector_model)

def load_recognition_model():
    recognizer_model = "nn4.small2.v1.t7"
    return cv2.dnn.readNetFromTorch(recognizer_model)

def detect_faces(frame, net):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    return detections, h, w

def get_face_blob(face, recognition_net):
    face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    recognition_net.setInput(face_blob)
    return recognition_net.forward().flatten()
