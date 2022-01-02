import numpy as np
import argparse
import imutils
import glob
import cv2
from numpy.lib.type_check import real_if_close

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, default="./images/cod",
                help="path to input images")
ap.add_argument("-t", "--template", type=str, default="./images/cod_logo.png",
                help="path to template images")
args = ap.parse_args()

template = cv2.imread(args.template)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)

for imagePath in glob.glob(args.images + "/*.jpg"):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        ratio = gray.shape[1] / float(resized.shape[1])

        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, ratio)

    (_, maxLoc, ratio) = found
    (startX, startY) = (int(maxLoc[0] * ratio),
                        int(maxLoc[1] * ratio))
    (endX, endY) = (int((maxLoc[0] + tW) * ratio),
                    int((maxLoc[1] + tH) * ratio))

    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
