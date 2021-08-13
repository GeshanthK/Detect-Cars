import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
file_name = input("Enter file (ie.cars.jpg): ")
im = cv2.imread(file_name)
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)
plt.imshow(output_image)
print('Number of cars in the image is '+ str(label.count('car')))
plt.show()