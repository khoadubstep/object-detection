from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascades", type=str, default="./cascades",
                help="path to input directory containing haar cascades")
args = ap.parse_args()

detectorPaths = {
    "face": "haarcascade_face.xml",
    "eye": "haarcascade_eye.xml",
    "mounth": "haarcascade_mounth.xml",
}

print("[INFO] loading haar cascades...")
detectors = {}

for (name, path) in detectorPaths.items():
    path = os.path.sep.join([args.cascades, path])
    detectors[name] = cv2.CascadeClassifier(path)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = detectors["face"].detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    for (fX, fY, fW, fH) in faceRects:
        faceROI = gray[fY:fY + fH, fX:fX + fW]

        eyeRects = detectors["eye"].detectMultiScale(
            faceROI, scaleFactor=1.1, minNeighbors=10,
            minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)

        mounthRects = detectors["mounth"].detectMultiScale(
            faceROI, scaleFactor=1.1, minNeighbors=10,
            minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)

        for (eX, eY, eW, eH) in eyeRects:
            ptA = (fX + eX, fY + eY)
            ptB = (fX + eX + eW, fY + eY + eH)
            cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)

        for (sX, sY, sW, sH) in mounthRects:
            ptA = (fX + sX, fY + sY)
            ptB = (fX + sX + sW, fY + sY + sH)
            cv2.rectangle(frame, ptA, ptB, (255, 0, 0), 2)

        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                      (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
