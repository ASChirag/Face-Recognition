import cv2
import numpy as np
from scipy.spatial import distance
from utils import load_detection_model, load_recognition_model, detect_faces, get_face_blob
from face_database import load_face_database
import mediapipe as mp

# def recognize_faces(frame, detection_net, recognition_net, known_embeddings, known_names):
#     detections, h, w = detect_faces(frame, detection_net)

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.8:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             face = frame[startY:endY, startX:endX]

#             vec = get_face_blob(face, recognition_net)

#             name = "Unknown"
#             min_dist = float("inf")
#             for j, known_embedding in enumerate(known_embeddings):
#                 dist = distance.euclidean(vec, known_embedding)
#                 if min_dist < 0.8:
#                     min_dist = dist
#                     name = known_names[j]

#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#             cv2.putText(frame, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     return frame
def recognize_faces(frame, detection_net, recognition_net, known_embeddings, known_names):
    detections, h, w = detect_faces(frame, detection_net)
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.8:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            # Debugging: Save the face image to check
            face_image_path = 'detected_face.jpg'
            cv2.imwrite(face_image_path, face)
            
            vec = get_face_blob(face, recognition_net)

            name = "Unknown"
            min_dist = float("inf")
            threshold = 1.3 # Set your distance threshold here

            for j, known_embedding in enumerate(known_embeddings):
                dist = distance.euclidean(vec, known_embedding)
                if dist < min_dist:
                    min_dist = dist
                    name = known_names[j]

            if min_dist >= threshold:
                name = "Unknown"

            # Debugging: Print distance and name
            print(f"Detected face distance: {min_dist}, Recognized name: {name}")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


def detect_body(frame,mp_pose,pose,mp_drawing,frame_height):
    results = pose.process(frame)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_hip_y = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame_height)
        right_hip_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame_height)

        mp_drawing.draw_landmarks(frame,results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if left_hip_y > 200 or right_hip_y > 200:
            print("Chirag")
    return frame

