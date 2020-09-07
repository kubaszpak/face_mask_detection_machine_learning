import os
from xml.etree import ElementTree
from PIL import Image
import numpy as np
import time
import cv2

project_dir = os.path.dirname(os.path.abspath(__file__))
dataset = 'dataset'
annotations_dir_name = 'annotations'
images_dir_name = 'images'

annotations_full_dir_name = os.path.join(project_dir, dataset, annotations_dir_name)
images_full_dir_name = os.path.join(project_dir, dataset, images_dir_name)


def get_box(face):
    
    coordinates_section = face.find('bndbox')

    xmin = int(coordinates_section.find('xmin').text)
    ymin = int(coordinates_section.find('ymin').text)
    xmax = int(coordinates_section.find('xmax').text)
    ymax = int(coordinates_section.find('ymax').text)
    
    return [ymin, ymax, xmin, xmax]

def get_category(face):

    return face.find('name').text

def main():
    face_images = []
    face_labels = []
    for root, subFolders, files in os.walk(annotations_full_dir_name):
        for file in files:
            if file.endswith('.xml'):
                full_file_path = os.path.join(root,file)
                dom = ElementTree.parse(full_file_path)
                picture_filename = dom.find('filename').text # We expect every annotation to be connected to one png file

                img_path = os.path.join(images_full_dir_name, picture_filename)
                # print(img_path)

                pil_image = Image.open(img_path) # RGB

                image_array = np.asarray(pil_image, 'uint8')

                # pil_image = Image.open(img_path).convert('RGB')
                # im = cv2.imread(img_path) # BGR

                faces = dom.findall('object')
                
                for face in faces:
                    coordinates = get_box(face)
                    category = get_category(face)
                    # print(category, coordinates)
                    roi = image_array[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]]
                    # im = Image.fromarray((roi * 255).astype(np.uint8))
                    size = (224,224)
                    final_image = np.resize(roi, size)
                    # final_image.show()
                    # res = cv2.resize(roi, dsize=size)
                    # cv2.imshow('Window',res)
                    # cv2.waitKey(0)
                    # final_image.show()

                break
            
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(end - start)

