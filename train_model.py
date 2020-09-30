import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
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
from read_dataset1 import read_dataset1
from read_dataset2 import read_dataset2

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

INIT_LR = 1e-4
EPOCHS = 10
BS = 32

def train_model(face_images, face_labels):

    # print(len(face_images), len(face_labels))
    face_images = np.array(face_images, dtype='float32')
    # print(face_images.shape)
    # print("face",face_images)
    face_labels = np.array(face_labels)

    # exit()

    lb = LabelEncoder()
    face_labels = lb.fit_transform(face_labels)
    face_labels = to_categorical(face_labels)

    # print(face_labels)

    aug = ImageDataGenerator(
        zoom_range=0.1,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    baseModel = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    #  input_tensor = Input(shape=(224, 224, 3))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.25)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    # print(model.input.shape)

    for layer in baseModel.layers:
        layer.trainable = False

    model.summary()

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


def main():
    face_images, face_labels = read_dataset2()
    train_model(face_images, face_labels)

if __name__ == "__main__":
    # start = time.time()
    main()
    # end = time.time()
    # print('Execution time:', end - start)
