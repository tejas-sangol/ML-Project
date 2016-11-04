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

def run(j):
    k_value =j;
    svm = SVM.SVC(kernel='linear',C=100);		        #Untrained SVM Classifier
    rf = RandomForestClassifier();      #Untrained Random Forest Classifier
    mlp = MLPClassifier();              #Untrained Neural network Classifier
    etc = ExtraTreesClassifier();       #Untrained ExtraTrees Classifier

    error=[]





    test_postive_features=[];
    test_negative_features=[];
    postive_features =[]    # Will be filled with the HOG descriptors of the cropped images(128 x 128 pixels).
    negative_features =[];  # Will be filled with hard negatives from the uncropped images.
    for i in range(1,6):
    	for root,_,files in os.walk('./training_data/cropped/'+str(i)):
    		for file in files:
                 if j==i:
                    test_postive_features.append(feature.hog(color.rgb2grey(io.imread(os.path.join(root,file)))));

                 else:
                    postive_features.append(feature.hog(color.rgb2grey(io.imread(os.path.join(root,file)))));



    # Filling the "negative_features" array with HOG fields of the negative sub-images from the uncropped images.


    pos_of_hand = open('./training_data/bounding_boxes.csv').read().split('\n')[1:];
    for k in range(1000):

    	prop = pos_of_hand[k].split(',')
    	image = io.imread('./training_data/raw/' + prop[5] + '/' +prop[0]);


    	y,x,_= image.shape

    	initial_shift=10;
    	for i in range(int(prop[1])+initial_shift,x-128,initial_shift):
    		for j in range(int(prop[2])+initial_shift,y-128,initial_shift):
    			# if (i,j)==(int(prop[1]),int(prop[2])): continue;
    			if (k%200)+1 == j: test_negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128))));
                else: negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128))));


    	for i in range(min(int(prop[3])-initial_shift,x),128,-initial_shift):
    		for j in range(min(int(prop[4])-initial_shift,y),128,-initial_shift):
    			if (i,j)==(int(prop[3]),int(prop[4])): continue;
    			if (k%200)+1 == j: test_negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i-128,i,j-128,j))));
                else: negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i-128,i,j-128,j))));

    	if len(negative_features) >=1000: break;



    X = postive_features + negative_features;
    print len(postive_features),len(negative_features)
    print len(test_postive_features),len(test_negative_features)
    Y = [1]*len(postive_features) + [0]*len(negative_features);
    # svm.fit(X,Y);
    # rf.fit(X,Y);
    mlp.fit(X,Y);
    # etc.fit(X,Y);
    test_features = test_postive_features + test_negative_features;
    test_classes = [1]*len(test_postive_features) + [0]*len(test_negative_features)
    error.append(mlp.score(test_features,test_classes));
    print k_value,error

p=Pool(5);
p.map(run,[i for i in range(1,6)])


#Dump the trained classifiers into into the dump folder
# with open('./dumps/svm','wb') as d:
# 	import pickle
# 	pickle.dump(svm,d);
# 	pickle.dump(rf,d);
# 	pickle.dump(mlp,d);
# 	pickle.dump(etc,d);
