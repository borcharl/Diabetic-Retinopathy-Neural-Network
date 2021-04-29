# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:58:07 2021

@author: DREAM_Student
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

train = ImageDataGenerator(rescale = 1/255)
train_dataset = train.flow_from_directory('C:\\Users\\DREAM_Student\\Desktop\\Training',
                                          target_size = (600,600),
                                          batch_size = 3,
                                          class_mode = 'binary')
validation_dataset = train.flow_from_directory('c:\\Users\\DREAM_Student\\Desktop\\Validation',
                                          target_size = (600,600),
                                          batch_size = 3,
                                          class_mode = 'binary')
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation = 'relu', input_shape = (600,600,3)),
                                    tf.keras.layers.AvgPool2D(2,2),
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (600,600,3)),
                                    tf.keras.layers.AvgPool2D(2,2),
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', input_shape = (600,600,3)),
                                    tf.keras.layers.AvgPool2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512,activation = 'relu'),
                                    tf.keras.layers.Dense(1,activation = 'sigmoid'),
                                    ])

model.compile(loss = 'binary_crossentropy',
              optimizer = RMSprop(lr=0.001),
              metrics = ['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch = 3,
                      epochs = 20,
                      validation_data = validation_dataset)

trCat = ["class1", "class4"]
dir_path = 'C:/Users/DREAM_Student/Desktop/Training'

for category in trCat:
    path = os.path.join(dir_path, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        
        plt.imshow(img_array, cmap='gray')
        plt.show()
        
        x= image.img_to_array(img_array)
        x= np.expand_dims(x,axis = 0)
        images= np.vstack([x])
        val= model.predict(images)
        if val==0:
            print('class 1')
        else:
            print('class 4')
            
for epoch in range():
    correct = 0
    for i, (inputs,labels) in enumerate (train_dataset):
        ...
        output = len(inputs)
        ...

        correct += (output == labels).float().sum()

    accuracy = 100 * correct / len(train_dataset)
    # trainset, not train_loader
    # probably x in your case

    print("Accuracy = {}".format(accuracy))
model.summary()

