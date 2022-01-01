from halo.detection import ShapeDetector
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="./images/shapes.png",
                help="path to the input image")
args = ap.parse_args()

image = cv2.imread(args.image)
resized = imutils.resize(image, width=300)
scale = image.shape[0] / float(resized.shape[0])

gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
shapeDetector = ShapeDetector()

for contour in contours:
    M = cv2.moments(contour)
    centerX = int((M["m10"] / M["m00"]) * scale)
    centerY = int((M["m01"] / M["m00"]) * scale)
    shape = shapeDetector.detect(contour)

    contour = (contour.astype("float") * scale).astype("int")
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
