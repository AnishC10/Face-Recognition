import face_recognition
import cv2
import os
import matplotlib.pyplot as plt

known_faces_file = "KNOWN DATA HERE"
unknown_faces_file = "UNKNOWN DATA HERE"
tolerance = 0.6
model = "cnn"

known_face_encodings = []

known_names =[]

#Create known face encodings and names
for name in os.listdir(known_faces_file):
    for filename in os.listdir(f"{known_faces_file}/{name}"):
        image = face_recognition.load_image_file(f"{known_faces_file}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)
        if len(encoding) > 0: 
            known_face_encodings.append(encoding[0]) 
            known_names.append(name)

#Identify unknown faces
for filename in os.listdir(unknown_faces_file):
    image = face_recognition.load_image_file(f"{unknown_faces_file}/{filename}")
    locations = face_recognition.face_locations(image, model = model)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for encoding, location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_face_encodings, encoding, tolerance = tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            top_left = (location[3], location[0])
            bottom_right = (location[1], location[2])
            color = [0,255,0]
            cv2.rectangle(image, top_left, bottom_right, color, 4)
            top_left = (location[3], location[2])
            bottom_right = (location[1], location[2]+23)
            color = [0,255,0]
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (location[3] + 100, location[2]+16), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 6)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
