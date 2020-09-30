import cv2
from xml.etree import ElementTree
import os
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

project_dir = os.path.dirname(os.path.abspath(__file__))
dataset = 'dataset'
annotations_dir_name = 'annotations'
images_dir_name = 'images'

annotations_full_dir_name = os.path.join(
    project_dir, dataset, annotations_dir_name)
images_full_dir_name = os.path.join(project_dir, dataset, images_dir_name)
# categories = ["without_mask","with_mask","mask_weared_incorrect"]

def get_box(face):

    coordinates_section = face.find('bndbox')

    xmin = int(coordinates_section.find('xmin').text)
    ymin = int(coordinates_section.find('ymin').text)
    xmax = int(coordinates_section.find('xmax').text)
    ymax = int(coordinates_section.find('ymax').text)

    return [ymin, ymax, xmin, xmax]


def get_category(face):

    return face.find('name').text


def read_dataset1():
    face_images = []
    face_labels = []
    for root, subFolders, files in os.walk(annotations_full_dir_name):
        for file in files:
            if file.endswith('.xml'):
                full_file_path = os.path.join(root, file)
                dom = ElementTree.parse(full_file_path)
                # We expect every annotation to be connected to one png file
                picture_filename = dom.find('filename').text

                img_path = os.path.join(images_full_dir_name, picture_filename)
                # print(img_path)

                # pil_image = Image.open(img_path)  # RGB
                # image_array = np.asarray(pil_image, 'uint8')

                image_array = cv2.imread(img_path)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

                # image = tf.keras.preprocessing.image.load_img(img_path)
                # image_array = keras.preprocessing.image.img_to_array(
                #     image, dtype='uint8')

                faces = dom.findall('object')

                for face in faces:
                    coordinates = get_box(face)
                    category = get_category(face)
                    # print(category, coordinates)
                    final_image = image_array[coordinates[0]:coordinates[1],
                                              coordinates[2]:coordinates[3]]
                    # im = Image.fromarray((final_image * 255).astype(np.uint8))
                    size = (224, 224)
                    final_image = cv2.resize(final_image, dsize=size)
                    final_image = preprocess_input(final_image)
                    # final_image = np.resize(final_image, size)
                    # final_image = img_to_array(final_image)
                    # print("to array",final_image)
                    # print("preprocessing",final_image)
                    # final_image.show()
                    # cv2.imshow('Window',final_image)
                    # cv2.waitKey(0)
                    # final_image.show()
                    face_images.append(final_image)
                    face_labels.append(category)
                    # print(final_image.shape)

                # break

    return face_images, face_labels
