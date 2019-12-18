from scipy.spatial import distance as dist
from imutils import face_utils
import playsound
import imutils
import dlib
import cv2
#to capture the frame we use video capture (0) for laptop camerA
cap = cv2.VideoCapture(0)

#Dlib --- use to find facial landmarks
detector = dlib.get_frontal_face_detector()
#to predict the 68 land marks on the face
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
COUNTER = 0
ALARM_ON = False

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates

    #there are 6 point on each eye we are calcluting the euclidian distance b/w vertical and horizontal line
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#it gives you the facial landmark of left and right eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


#face utils is use to find out facial landmark of particular part of the face

#setting threshold value
EYE_AR_THRESH = 0.3

#if the frame captured are more the 48 almarm goes on
EYE_AR_CONSEC_FRAMES = 48

while True:
    #reading the image using cap.read
    _, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        #converting it into numpy array
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        #convex hull is used to draw a convex shaoe around you eye
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            #if ear is less the the threshold the count++ else count=0
            #and once count become more the 48 (u can chose you own threshold) alarm goes out
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True
                    # sound played in the background
                    playsound.playsound('alarm.wav',True)
            else:
                pass
        else:
            COUNTER = 0
            ALARM_ON = False


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()



