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
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K

from sknn.mlp import Convolution

batch_size = 100
nb_classes = 24
nb_epoch = 2000

hash={};
hash['A']=0;
hash['B']=1;
hash['C']=2;
hash['D']=3;
hash['E']=4;
hash['F']=5;
hash['G']=6;
hash['H']=7;
hash['I']=8;
hash['K']=9;
hash['L']=10;
hash['M']=11;
hash['N']=12;
hash['O']=13
hash['P']=14
hash['Q']=15
hash['R']=16
hash['S']=17
hash['T']=18
hash['U']=19
hash['V']=20
hash['W']=21
hash['X']=22
hash['Y']=23


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
    for dir in directories[:2]:
        if dir[0]=='.' : continue;
        print(dir)
        with open('./dataset/'+dir+'/'+dir+'_loc.csv') as file_list:
            file_list = file_list.read().split('\n')[1:];
            for entry in file_list:
                if len(entry)==0:continue;
                symbol = entry[entry.find('/')+1:entry.find('/')+3];


                image =io.imread('./dataset/'+ entry.split(',')[0]);
                # image = feature.hog(color.rgb2grey(image));

                # image = transform.resize(image,(200,200));
                if int(symbol[1])>=8:
                    X_test.append(image);
                    y_test.append(hash[symbol[0]]);
                else:
                    X_train.append(image);
                    y_train.append(hash[symbol[0]]);


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


mlp = Convolution(type='Rectifier',
                    kernel_shape=(240,320,3),
                    kernel_stride=(2,2),
                    border='full',
                    pool_shape=(2,2),
                    pool_type='mean',
                    scale_factor=(2,2),
                    dropout=0.25,
                    normalize='weight',
                    )

mlp.fit(X_train,X_test);

print(mlp.score(X_test,Y_test));
