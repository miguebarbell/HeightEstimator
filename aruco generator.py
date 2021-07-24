import cv2
import numpy as np


# todo: documentation
# generate aruco dict
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

# generate the space 300x300
tag = np.zeros((300, 300, 1), dtype="uint8")

# draw the tag id=8 in the space
"""
The first parameter is the Dictionary object previously created.
The second parameter is the marker id, in this case the marker 68 of the dictionary DICT_ARUCO_ORIGINAL. 
The third parameter, 300, is the size of the output marker image (300x300). So, for instance, you cannot generate an image of 5x5 pixels for a marker size of 6x6 bits.
The fourth parameter is the output image.
Finally, the last parameter is an optional parameter to specify the width (internal bits) of the marker black border. The default value is 1.
"""
cv2.aruco.drawMarker(aruco_dict, 68, 300, tag, 1)

# show the image
cv2.imshow('tag', tag)
cv2.waitKey(0)

# ask for saving the image
output = "tag.jpg"
print('Save the image? [Y]/n')
answer = input()
if answer != 'n':
    cv2.imwrite(f'images/{output}', tag)
    print(f'Tag saved as {output}')
else:
    print('Tag not saved, quitting...')
