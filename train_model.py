import os
from xml.etree import ElementTree
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import time
import cv2

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


def main():
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
                    # final_image = np.resize(final_image, size)
                    # final_image = img_to_array(final_image)
                    # print("to array",final_image)
                    # print("preprocessing",final_image)
                    # final_image.show()
                    final_image = cv2.resize(final_image, dsize=size)
                    final_image = preprocess_input(final_image)
                    # cv2.imshow('Window',final_image)
                    # cv2.waitKey(0)
                    # final_image.show()
                    face_images.append(final_image)
                    face_labels.append(category)
                    # print(final_image.shape)

                # break

    # print(len(face_images), len(face_labels))
    face_images = np.array(face_images, dtype='float32')
    print(face_images.shape)
    # print("face",face_images)
    face_labels = np.array(face_labels)

    # exit()

    lb = LabelEncoder()
    face_labels = lb.fit_transform(face_labels)
    face_labels = to_categorical(face_labels)

    aug = ImageDataGenerator(
        zoom_range=0.1,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

    #  input_tensor = Input(shape=(224, 224, 3))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.25)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    print(model.input.shape)

    for layer in baseModel.layers:
        layer.trainable = False

    model.summary()

    INIT_LR = 1e-4
    EPOCHS = 10
    BS = 32

    (train_x, test_x, train_y, test_y) = train_test_split(face_images,
                                                          face_labels, test_size=0.2, stratify=face_labels, random_state=42)

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # print(model.input.shape)

    H = model.fit(
        aug.flow(train_x, train_y, batch_size=BS),
        steps_per_epoch=len(train_x) // BS,
        validation_data=(test_x, test_y),
        validation_steps=len(test_x) // BS,
        epochs=EPOCHS)

    predIdxs = model.predict(test_x, batch_size=BS)

    predIdxs = np.argmax(predIdxs, axis=1)

    print(classification_report(test_y.argmax(axis=1), predIdxs,
                                target_names=lb.classes_))

    model.save('model2.h5')


if __name__ == "__main__":
    # start = time.time()
    main()
    # end = time.time()
    # print('Execution time:', end - start)
