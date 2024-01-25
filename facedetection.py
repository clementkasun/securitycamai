# import cv2

# face_classifier = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# video_capture = cv2.VideoCapture(0)

# def detect_bounding_box(vid):
#     gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
#     for (x, y, w, h) in faces:
#         cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
#     return faces

# while True:
#     result, video_frame = video_capture.read()  # read frames from the video
#     if result is False:
#         break  # terminate the loop if the frame is not read successfully

#     faces = detect_bounding_box(
#         video_frame
#     )  # apply the function we created to the video frame

#     cv2.imshow(
#         "My Face Detection Project", video_frame
#     )  # display the processed frame in a window named "My Face Detection Project"

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# video_capture.release()
# cv2.destroyAllWindows()

import cv2
import face_recognition

# Load the image whose face you want to compare
known_face_image = face_recognition.load_image_file("./face1.jpg")
known_face_encoding = face_recognition.face_encodings(known_face_image)[0]

video_capture = cv2.VideoCapture(0)

while True:
    result, video_frame = video_capture.read()

    if result is False:
        break

    # Find face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(video_frame)
    face_encodings = face_recognition.face_encodings(video_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the detected face with the known face
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

        name = "Unknown"
        if matches[0]:
            name = "Known Person"

        # Draw a rectangle around the face and display the name
        cv2.rectangle(video_frame, (left, top), (right, bottom), (0, 255, 0), 4)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(video_frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow("Face Recognition", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
