#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import all 


# In[33]:


import warnings
warnings.filterwarnings('ignore')


# In[76]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import cv2
import tensorflow as tf
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import ImageFile
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[4]:


import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# In[ ]:


#load dataset


# In[89]:


datagen= ImageDataGenerator()
train_it= datagen.flow_from_directory('data/')


# In[36]:


CLASS_NAMES = ['Blue', 'Green','Red']


# In[37]:


num_red = len(os.listdir('data/train/red'))
print(num_red)
num_blue = len(os.listdir('data/train/blue'))
print(num_blue)
num_green= len(os.listdir('data/train/green'))
print(num_green)
train_dir = os.path.join(r'C:\Users\HP\data', 'train')
print(train_dir)
valid_dir= os.path.join(r'C:\Users\HP\data','validation')
print(valid_dir)
test_dir=os.path.join(r'C:\Users\HP\data','testing')
print(test_dir)


# In[ ]:


#Pipeline


# In[38]:


BATCH_SIZE = 32
IMG_SHAPE  = 64

image_gen = ImageDataGenerator(rescale=1./255)

one_image = image_gen.flow_from_directory(directory=train_dir,
                                          batch_size=BATCH_SIZE,
                                          classes  = CLASS_NAMES ,
                                          shuffle=True,
                                          target_size=(IMG_SHAPE,IMG_SHAPE),
                                          class_mode='categorical')

test_image = image_gen.flow_from_directory(directory=valid_dir,
                                          batch_size=BATCH_SIZE,
                                          classes  = CLASS_NAMES ,
                                          shuffle=True,
                                          target_size=(IMG_SHAPE,IMG_SHAPE),
                                          class_mode='categorical')



plt.imshow(one_image[0][0][0])
plt.show()
print(one_image)


# In[55]:


#Buid the model


# In[17]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


# In[ ]:


#model summary


# In[18]:


model.summary()


# In[ ]:


#compile model


# In[19]:



opt = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])


# In[ ]:


#train the model


# In[20]:


model.fit_generator(one_image,
                                    steps_per_epoch = 2700,
                                    epochs = 25,
                                    validation_data = test_image,
                                    validation_steps = 300)


# In[42]:


test_loss, test_accuracy = model.evaluate(test_image, verbose=1)
print("Loss  : ", test_loss)
print("Accuracy  :",test_accuracy*100)


# In[ ]:


# test the image ....


# In[86]:


Image=image.load_img(r'C:\Users\HP\data\testing\26.jpg', target_size = (64,64))
Image = np.expand_dims(Image, axis=0)


# In[87]:


predictions = model.predict(Image)

print(predictions)


# In[ ]:


# predict the output...


# In[88]:



if predictions[0][0] == 1:
    print("Blue")
elif predictions[0][1] == 1:
    print("Green")
elif predictions[0][2] == 1:
    print("Red")

