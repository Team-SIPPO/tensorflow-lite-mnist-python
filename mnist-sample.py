import tensorflow as tf
import cv2
import os
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# create model architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(784, )))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

# learning
model.fit(x_train, y_train, epochs=10, batch_size=32)
# save model
model.save("keras_model.h5", overwrite=True)

# model save and transform
converter = tf.lite.TFLiteConverter.from_keras_model_file("keras_model.h5")
tflite_model = converter.convert()
open("converted_keras_model.tflite", "wb").write(tflite_model)


def save_images():
    image_save_dir = "./images"
    if not os.path.exists(image_save_dir):
        os.mkdir(image_save_dir, )

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    image_num = 10
    for i in np.arange(0, image_num):
        image = x_test[i]
        image_file = image_save_dir + "/" + str(i) + ".bmp"
        cv2.imwrite(image_file, image)


