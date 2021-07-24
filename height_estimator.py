import cv2
import imutils
import numpy as np
from datetime import datetime

# todo: put an icon in the aruco tag
# import image
input_image = "images/test.jpg"
image = cv2.imread(input_image)
image = imutils.resize(image, width=1200)
# detect aruco tag
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
aruco_params = cv2.aruco.DetectorParameters_create()
(corners, aruco_id, rejected) = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
if aruco_id != None:
    print(f"[INFO] Found aruco tag id: {np.squeeze(aruco_id)}")
    corners = np.squeeze(corners[0])
    lowest_point = int(max(corners[2:4, 1]))
    # print(f"[INFO] Lowest point: {lowest_point}")
    # cv2.line(image, (610, lowest_point), (529, lowest_point), (0, 255, 0), 3)
    # calcular la altura del tag
    tag_height = corners[2, 1] - corners[1, 1]

# detect face
cascade = "../../commonfiles/haarcascade_frontalface_default.xml"
# cascade = "../../commonfiles/haarcascade_fullbody.xml"
detector = cv2.CascadeClassifier(cascade)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("[INFO] performing Human detection...")
rects = detector.detectMultiScale(gray, scaleFactor=1.2,
                                  minNeighbors=5, minSize=(30, 30),
                                  flags=cv2.CASCADE_SCALE_IMAGE)
print("[INFO] {} Human detected...".format(len(rects)))

# line parameters
thickness = 1
line_color = (0, 255, 0)
# loop over the bounding boxes
for (x, higher_point, w, h) in rects:
    # draw the line of the total height
    cv2.rectangle(image, (x+w+6, higher_point-13), (x+w+95, higher_point+3), (0, 0, 0), -1)
    cv2.line(image, (x+w+5, lowest_point), (x+w+5, higher_point), line_color, thickness)
    cv2.line(image, (x+w+5, lowest_point), (x+w-10, lowest_point), line_color, thickness)
    cv2.line(image, (x+w+5, higher_point), (x+w-10, higher_point), line_color, thickness)

    # compute distances
    human_height = (((lowest_point - higher_point)*19)/tag_height) + 5
    cv2.putText(image, f"{str(round(human_height, 1))} cms", (x+w+7, higher_point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, thickness)

# Syntax: cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])

# draw the image using the time
cv2.imshow('test', image)
cv2.waitKey(0)

# save the result
# print("Do you wan to save the image? [Y]/n")
answer = input("Do you wan to save the image? [Y]/n\n")
if answer.lower() != 'n' or answer.lower() != 'no':
    output = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.jpg"
    cv2.imwrite(output, image)
