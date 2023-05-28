from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from PIL import Image
from numpy import array


local_zip = '/content/USPS_images.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/trainntest')

train_dir = '/content/trainntest/train'
validation_dir = '/content/trainntest/test'
y_train = []
y_test = []
for path in os.listdir(train_dir):
    if os.path.isfile(os.path.join(train_dir, path)):
        y_train.append(int(path[0]))

for path in os.listdir(validation_dir):
  if os.path.isfile(os.path.join(validation_dir, path)):
      y_test.append(int(path[0]))

x_train = []
for path in os.listdir(train_dir):
  if os.path.isfile(os.path.join(train_dir,path)):
    x_train.append(cv2.cvtColor(cv2.imread(f"{train_dir}/{path}"), cv2.COLOR_RGB2GRAY))

x_test = []
for path in os.listdir(validation_dir):
  if os.path.isfile(os.path.join(validation_dir,path)):
    x_test.append(cv2.cvtColor(cv2.imread(f"{validation_dir}/{path}"), cv2.COLOR_RGB2GRAY))

x_train = np.array(x_train)
x_test = np.array(x_test)

num_classes = 10

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

y_train_cat = np.array(y_train_cat)
y_test_cat = np.array(y_test_cat)

x_train_final = x_train.reshape(-1 ,16*16) / 255
x_test_final = x_test.reshape(-1 ,16*16) / 255




from keras.utils import to_categorical
num_classes = 10

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(y_train_cat.shape)
print(y_test_cat.shape)

print(y_train[0]) # 5 >>>> [0,0,0,0,0,1,0,0,0,0]
print(y_train_cat[0])

from keras.layers import Dense, Input
from keras.models import Sequential

model = Sequential()
model.add(Input(shape = (16*16)))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(num_classes , activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

# model.summary()

batch_size = 128
epochs = 30
model.fit(x_train_final, y_train_cat,
          batch_size= batch_size ,
          epochs=epochs, verbose= 1,
          validation_data=(x_test_final,y_test_cat))

import numpy as np
from google.colab import files
import keras.utils as image

uploaded = files.upload()

for fn in uploaded.keys():
    # predicting images
    path = '/content/' + fn
    img = image.load_img(path, target_size=(16,16))
    x = image.img_to_array(img)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = x.reshape(16*16) 
    x = np.expand_dims(x, axis=0) / 255.
    classes = model.predict(x, batch_size=10)
    print(np.argmax(classes))