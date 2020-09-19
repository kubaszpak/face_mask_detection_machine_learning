import cv2
import tensorflow.keras as keras
import numpy as np

categories = ["mask_weared_incorrect", "with_mask", "without_mask"]

cascade_location = 'cascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_location)

cap = cv2.VideoCapture(0)
cap.set(3, 400)
cap.set(4, 600)
cap.set(10, 100)

model = keras.models.load_model('model1.h5')

# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(loss="categorical_crossentropy", optimizer=opt,
#               metrics=["accuracy"])

# # train the head of the network
# H = model.fit(
#     aug.flow(trainX, trainY, batch_size=BS),
#     steps_per_epoch=len(trainX) // BS,
#     validation_data=(testX, testY),
#     validation_steps=len(testX) // BS,
#     epochs=EPOCHS,
#     class_weight={0: 5, 1: 1, 2: 10})

while True:

    success, img = cap.read()
    # img = cv2.imread('dataset/images/maksssksksss0.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 3)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = img[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, dsize=(224, 224))
        roi_color = keras.applications.mobilenet_v2.preprocess_input(roi_color)
        # roi_gray = imgGray[y:y+h, x:x+w]
        list_of_images = []
        list_of_images.append(roi_color)
        predictions = model.predict(np.asarray(list_of_images)[0:1])
        predictions = np.argmax(predictions, axis=1)
        print(categories[predictions[0]])
        break

    cv2.imshow('Video', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
