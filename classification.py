import warnings

warnings.filterwarnings('ignore')
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
image = mpimg.imread('C:/Users/HP/data/testing/1.jpg');

Image = image.reshape([-1,225, 225, 3])

datagen = ImageDataGenerator()
train_it = datagen.flow_from_directory('C:/Users/HP/data/')

num_red = len(os.listdir('C:/Users/HP/data/train/red'))
print(num_red)
num_blue = len(os.listdir('C:/Users/HP/data/train/blue'))
print(num_blue)
num_green = len(os.listdir('C:/Users/HP/data/train/green'))
print(num_green)
train_dir = os.path.join(r'C:\Users\HP\data', 'train')
print(train_dir)
valid_dir = os.path.join(r'C:\Users\HP\data', 'validation')
print(valid_dir)
test_dir = os.path.join(r'C:\Users\HP\data', 'testing')
print(test_dir)

BATCH_SIZE = 10
IMG_SHAPE = 225

image_gen = ImageDataGenerator(rescale=1. / 255)

one_image = image_gen.flow_from_directory(directory=train_dir,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          target_size=(IMG_SHAPE, IMG_SHAPE),
                                          class_mode='binary')

plt.imshow(one_image[0][0][0])
plt.show()
print(one_image)

image_gen_train = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(directory=train_dir,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='binary')
print(train_data_gen)

y_true_labels = train_data_gen.classes
print(y_true_labels)

image_gen_val = ImageDataGenerator(rescale=1. / 255)

val_data_gen = image_gen_val.flow_from_directory(directory=valid_dir,
                                                 batch_size=BATCH_SIZE,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')
y_train_labels = val_data_gen.classes
print(y_train_labels)

layer_neurons = [1024, 512, 256, 128, 56, 28, 14]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(IMG_SHAPE, IMG_SHAPE, 3)))

for neurons in layer_neurons:
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 40

history = model.fit_generator(train_data_gen, epochs=EPOCHS,
                              validation_data=val_data_gen)

predictions = model.predict(Image)
classes = model.predict_classes(Image)
print(classes)

predictions[0]

classes = train_data_gen.class_indices
print(classes)

a = np.argmax(predictions[0])
if a == 0:
    print('Blue')
if a == 1:
    print('Green')
if a == 2:
    print('Red')
