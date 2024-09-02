import cv2
import numpy as np
from scipy.spatial import distance
from utils import load_detection_model, load_recognition_model, detect_faces, get_face_blob
from face_database import load_face_database

def recognize_faces(frame, detection_net, recognition_net, known_embeddings, known_names):
    detections, h, w = detect_faces(frame, detection_net)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            vec = get_face_blob(face, recognition_net)

            name = "Unknown"
            min_dist = float("inf")
            for j, known_embedding in enumerate(known_embeddings):
                dist = distance.euclidean(vec, known_embedding)
                if dist < min_dist:
                    min_dist = dist
                    name = known_names[j]

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame
