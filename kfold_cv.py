from sklearn import svm as SVM
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import skimage as ski
from skimage import io,transform,feature,color,data
import numpy as np
from sliding_window import sliding_window
import os
from multiprocessing import Pool
def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
    return arr[start_coloumn:end_coloumn,start_row:end_row];


def run(arg):
    j,C,gamma = arg
    k_value =j;
    svm = SVM.SVC(kernel='rbf',C=C,gamma=gamma);                #Untrained SVM Classifier
    rf = RandomForestClassifier();      #Untrained Random Forest Classifier
    mlp = MLPClassifier();              #Untrained Neural network Classifier
    etc = ExtraTreesClassifier();       #Untrained ExtraTrees Classifier

    postive_features =[]    # Will be filled with the HOG descriptors of the cropped images(128 x 128 pixels).
    negative_features =[];  # Will be filled with hard negatives from the uncropped images.
    positive_test_features =[];
    negative_test_features=[];
    #Filling the "postive_features" array with HOG fields.
    # Methods used ---
    # 1. io.imread() --> Loads the image as a 3D array from the file
    # 2. color.rgb2grey --> Normalizes the 3D coloured array to 2D greyscaled array necessary for HOG.
    # 3. features.hog --> Ruturns the HOG Descriptor for the 2D array

    for i in range(1,6):
        for root,_,files in os.walk('./training_data/cropped/'+str(i)):
            for file in (files[:40*j] + files[40*(j+1):]):
                 postive_features.append(feature.hog(color.rgb2grey(io.imread(os.path.join(root,file)))));
            for file in files[40*j:40*j + 40]:
                positive_test_features.append(feature.hog(color.rgb2grey(io.imread(os.path.join(root,file)))));

    # Filling the "negative_features" array with HOG fields of the negative sub-images from the uncropped images.


    pos_of_hand = open('./training_data/bounding_boxes.csv').read().split('\n')[1:];
    for k in range(1000):
        prop = pos_of_hand[k].split(',')
        image = io.imread('./training_data/raw/' + prop[5] + '/' +prop[0]);


        y,x,_= image.shape

        initial_shift=10;
        for i in range(int(prop[1])+initial_shift,x-128,initial_shift):
            for j in range(int(prop[2])+initial_shift,y-128,initial_shift):
                if (k%200)/40==(k_value-1) and len(negative_test_features) <=1000: negative_test_features.append(feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128))));
                elif len(negative_features) <=1000: negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128))));


        for i in range(min(int(prop[3])-initial_shift,x),128,-initial_shift):
            for j in range(min(int(prop[4])-initial_shift,y),128,-initial_shift):
                if (k%200)/40==(k_value-1)  and len(negative_test_features) <=1000: negative_test_features.append(feature.hog(color.rgb2grey(crop_image(image,i-128,i,j-128,j))));
                elif len(negative_features) <=1000:negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i-128,i,j-128,j))));

    X=postive_features + negative_features;
    Y=[1]*len(postive_features) + [0]*len(negative_features);
    svm.fit(X,Y);
    X2 = positive_test_features + negative_test_features;
    Y2=[1]*len(positive_test_features) + [0]*len(negative_test_features);
    error = svm.score(X2,Y2);
    print len(postive_features),len(negative_features);
    print len(positive_test_features),len(negative_test_features);
    print arg,error



p=Pool(4);
arg=[(i,C,gamma) for i in range(1,6) for C in [.001,.01,.1,1,10,100,1000] for gamma in [.001,.01,.1,1,10,100,1000]]
p.map(run,arg)


#Dump the trained classifiers into into the dump folder
# with open('./dumps/svm','wb') as d:
#     import pickle
#     pickle.dump(svm,d);
#     pickle.dump(rf,d);
#     pickle.dump(mlp,d);
#     pickle.dump(etc,d);
