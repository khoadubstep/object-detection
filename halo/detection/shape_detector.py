import cv2


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, contour):
        shape = "unidentified"
        perimeter = cv2.arcLength(contour, True)
        vertices = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        if len(vertices) == 3:
            shape = "triangle"

        elif len(vertices) == 4:
            (x, y, w, h) = cv2.boundingRect(vertices)
            aspectRatio = w / float(h)

            shape = "square" if aspectRatio >= 0.95 and aspectRatio <= 1.05 else "rectangle"

        elif len(vertices) == 5:
            shape = "pentagon"

        else:
            shape = "circle"

        return shape
