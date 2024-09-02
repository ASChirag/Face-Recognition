import os
import cv2
import numpy as np
import pickle

# Load the face recognition model
recognizer_model = "nn4.small2.v1.t7"  # Update this path
if not os.path.exists(recognizer_model):
    raise FileNotFoundError(f"Model file not found: {recognizer_model}")
recognition_net = cv2.dnn.readNetFromTorch(recognizer_model)

def get_face_embedding(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.imread(image_path)
    face_blob = cv2.dnn.blobFromImage(image, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    recognition_net.setInput(face_blob)
    vec = recognition_net.forward()
    return vec.flatten()

def create_face_database(known_faces_dir):
    if not os.path.isdir(known_faces_dir):
        raise FileNotFoundError(f"Known faces directory not found: {known_faces_dir}")
    
    known_embeddings = []
    known_names = []

    for file_name in os.listdir(known_faces_dir):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            name = os.path.splitext(file_name)[0]
            image_path = os.path.join(known_faces_dir, file_name)
            embedding = get_face_embedding(image_path)
            
            known_embeddings.append(embedding)
            known_names.append(name)

    with open("face_embeddings.pickle", "wb") as f:
        pickle.dump((known_embeddings, known_names), f)
    print("Face embeddings saved to face_embeddings.pickle")

def load_face_database():
    with open("face_embeddings.pickle", "rb") as f:
        return pickle.load(f)
    
if __name__ == "__main__":
    known_faces_dir = "Face-Recognition/Images"  # Update this path
    create_face_database(known_faces_dir)
