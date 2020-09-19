import cv2
import numpy as np

image = cv2.imread('dataset/images/maksssksksss0.png')
image = cv2.resize(image, dsize=(224, 224))
list_of_images = []
# list_of_images = np.empty([4,224,224,3],dtype='uint8')
# print(list_of_images.shape)
list_of_images.append(image)
# list_of_images = np.
# print(list_of_images.shape, list_of_images.size)
print(list_of_images)
