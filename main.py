import cv2
from utils import load_detection_model, load_recognition_model
from face_recognition import recognize_faces
from face_database import load_face_database

def main():
    detection_net = load_detection_model()
    recognition_net = load_recognition_model()
    known_embeddings, known_names = load_face_database()

    cap = cv2.VideoCapture(0)  # Replace 0 with the path to your video file if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = recognize_faces(frame, detection_net, recognition_net, known_embeddings, known_names)

        cv2.imshow("Face Recognition", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
