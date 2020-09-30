import os
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
dataset = 'dataset2'
# with_mask = os.path.join(project_dir,dataset,'with_mask')
# with_mask = os.path.join(project_dir,dataset,'without_mask')
dataset_dir = os.path.join(project_dir,dataset)


def read_dataset2():
    face_images = []
    face_labels = []
    for root, subfolders, files in os.walk(dataset_dir):
        # print(root,subfolders,files)
        for file in files:
            if(file.endswith('.jpg') or file.endswith('.png')):
                img_path = os.path.join(root,file)
                image_array = cv2.imread(img_path)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                category = os.path.basename(os.path.normpath(root)).replace(' ','_')
                size = (224, 224)
                image_array = cv2.resize(image_array,dsize=size)
                image_array = preprocess_input(image_array)
                face_images.append(image_array)
                face_labels.append(category)

    return face_images, face_labels



if __name__ == "__main__":
    face_images, face_labels = read_dataset2()
    for image in face_images:
        pass
        # print(image.shape, image.dtype)
    for label in face_labels:
        if label != 'with_mask' and label != 'without_mask':
            print('warning')
