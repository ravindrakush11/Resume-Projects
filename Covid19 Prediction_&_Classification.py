# -*- coding: utf-8 -*-
"""Covid19 Prediction & Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gVXSP8AYVmlJvuQ0nP7X-k3eTyXrxDOb
"""

from google.colab import drive
drive.mount('/content/drive')

from zipfile import ZipFile
file_name = "/content/drive/My Drive/CovidDataset.zip"
with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print("Completed")

from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
import tensorflow.keras as tf

#Training model(CNN)
model = Sequential()   ## creating a blank model
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))    ### reduce the overfitting
 
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())    ### input layer
model.add(Dense(64,activation='relu'))    ## hidden layer of ann
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))   ## output layer
 
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#Modlding train images
from tensorflow.keras.preprocessing import image
train_datagen = image.ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True)
 
test_dataset = image.ImageDataGenerator(rescale=1./255)

#Reshaping test and validation images 
train_generator = train_datagen.flow_from_directory(
    '/content/CovidDataset/Train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')
validation_generator = test_dataset.flow_from_directory(
    '/content/CovidDataset/Val',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')

#### Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=7,
    epochs = 20,
    validation_data = validation_generator,
    validation_steps=1
)

from tensorflow.keras.preprocessing import image
import numpy as np
img = image.load_img('/content/CovidDataset/Train/Covid/1-s2.0-S0140673620303706-fx1_lrg.jpg',target_size=(224,224))
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)   ### flattening
ypred = model.predict(img)
if ypred[0][0] == 1:
  print("Covid Negative")
else:
  print("Covid Positive")

ypred[0][0]

import pandas as pd
pd.DataFrame(history.history).plot(title="Train and validation results",figsize=(10,7));

loss, accuracy = model.evaluate(img, ypred)
print('Test accuracy:' , accuracy * 100,'%')

#Convert prediction probabilities into integers
y_preds = ypred.argmax(axis=1)

from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
#Confusion matrix

cm=confusion_matrix(ypred, ypred)
#Plot
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels='ypred')
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax);

#### save the model
model.save("covid_model.h5")

import tensorflow.keras as tf
mymodel = tf.models.load_model("/content/covid_model.h5")

from tensorflow.keras.preprocessing import image
import numpy as np
img = image.load_img('/content/CovidDataset/Train/Normal/IM-0221-0001.jpeg',target_size=(224,224))
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)   ### flattening
ypred = mymodel.predict(img)
if ypred[0][0] == 1:
  print("Covid Negative")
else:
  print("Covid Positive")


image = cv2.imread('/content/CovidDataset/Train/Normal/IM-0221-0001.jpeg')
cv2_imshow(image)


