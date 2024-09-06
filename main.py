import cv2
from utils import load_detection_model, load_recognition_model
from face_recognition import recognize_faces
from face_database import load_face_database
from face_recognition import detect_body
import numpy as np
import mediapipe as mp

def main():
    detection_net = load_detection_model()
    recognition_net = load_recognition_model()
    known_embeddings, known_names = load_face_database()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)  # Replace 0 with the path to your video file if needed
    frame_height = cv2.CAP_PROP_FRAME_HEIGHT
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame2 = cv2.line(frame, (200,480),(200,0), (124,56,200),2)
        output_frame = recognize_faces(frame, detection_net, recognition_net, known_embeddings, known_names)

        body_detect = detect_body(frame2,mp_pose,pose,mp_drawing,frame_height)
        both = np.concatenate((output_frame,body_detect), axis=1)
        cv2.imshow("Face Recognition", both)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
