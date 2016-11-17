'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import skimage as ski
from skimage import io,transform,feature,color,data
import pickle;
from multiprocessing import Pool;

import numpy as np
import os
np.random.seed(1337)  # for reproducibility

import keras.datasets.mnist as mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

batch_size = 100
nb_classes = 10
nb_epoch = 2000

# input image dimensions
img_rows, img_cols = 32, 32
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
X_train,X_test,y_train,y_test=[],[],[],[];
for root,directories,_ in os.walk('dataset'):
    for dir in directories:
        if dir[0]=='.' : continue;
        print(dir)
        with open('./dataset/'+dir+'/'+dir+'_loc.csv') as file_list:
            file_list = file_list.read().split('\n')[1:];
            for entry in file_list:
                if len(entry)==0:continue;
                symbol = entry[entry.find('/')+1:entry.find('/')+3];


                image =io.imread('./dataset/'+ entry.split(',')[0]);
                # image = transform.resize(image,(200,200));
                if int(symbol[1])>=8:
                    X_test.append(image);
                    y_test.append(symbol[1]);
                else:
                    X_train.append(image);
                    y_train.append(symbol[1]);

X_train = np.array(X_train);
X_test = np.array(X_test);
y_train = np.array(y_train);
y_test = np.array(y_test);

# import pickle;
# with open('cnn_data','wb') as data:
#     pickle.dump(main,data)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(5, 2, 2, input_shape=(240,320,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Convolution2D(5, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

# model.add(Convolution2D(5, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Convolution2D(5, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))



# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
