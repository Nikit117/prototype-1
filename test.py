import cv2 ,time
import winsound
from pygame import mixer
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# smile_cascade=cv2.CascadeClassifier("smile.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# haar_upper_body_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
palm_cascade = cv2.CascadeClassifier("palm.xml")
video=cv2.VideoCapture(0)
# address = "https://100.103.23.254:8080/video"
# video.open(address)
mixer.init()
sound = mixer.Sound('Alarm.wav')
KNOWN_DISTANCE = 30.2
KNOWN_WIDTH = 14.3 
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX
def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value
def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance
def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), WHITE, 1)
        face_width = w
    return face_width
ref_image = cv2.imread("Ref_image.png")
ref_image_face_width = face_data(ref_image)
focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width)
while True:
    check,frame=video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in face:
        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        face_width_in_frame = face_data(frame)
        if face_width_in_frame != 0:
            Distance = distance_finder(focal_length_found, KNOWN_WIDTH, face_width_in_frame)
            cv2.putText(frame, f"Distance = {round(Distance,2)} CM", (50, 50), fonts, 1, (WHITE), 2)
            if Distance <= 100:
                winsound.Beep(500,200)
                cv2.putText(frame,"You are coming too close", (50, 450), fonts, 1, (WHITE), 2)
        # smile=smile_cascade.detectMultiScale(gray,scaleFactor=1.8,minNeighbors=20)
        # for x,y,w,h in smile:
        #     img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        eyes = eye_cascade.detectMultiScale(img)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # upper_body = haar_upper_body_cascade.detectMultiScale(img)
        # for (x, y, w, h) in upper_body:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 68, 32), 1)
        #     cv2.putText(frame, "Upper Body Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 68, 32), 2)
        palm = palm_cascade.detectMultiScale(img)
        for (x, y, w,h) in palm:
            cv2.rectangle(frame, (x,y) , (x+w, y+h) , (12, 56, 56) , 2)
            cv2.putText(frame, "palm Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (12, 56, 56), 3)
            sound.play()
            cv2.putText(frame,"Suspicion Detected", (50, 400), fonts, 1, (WHITE), 2)
            # winsound.Beep(500,200)
    cv2.imshow('CAM',frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
         break

video.release()
cv2.destroyAllWindows()        