from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import skimage as ski
from skimage import io,transform,feature,color,data
import numpy as np
# import Image
import os

def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];



svm = svm.SVC(kernel='rbf')         #Untrained SVM Classifier
rf = RandomForestClassifier()       #Untrained Random Forest Classifier
mlp = MLPClassifier();              #Untrained Neural network Classifier
etc = ExtraTreesClassifier();       #Untrained ExtraTrees Classifier



postive_features =[]    # Will be filled with the HOG descriptors of the cropped images(128 x 128 pixels).
negative_features =[];  # Will be filled with hard negatives from the uncropped images.

#Filling the "postive_features" array with HOG fields.
# Methods used ---
# 1. io.imread() --> Loads the image as a 3D array from the file
# 2. color.rgb2grey --> Normalizes the 3D coloured array to 2D greyscaled array necessary for HOG.
# 3. features.hog --> Ruturns the HOG Descriptor for the 2D array
for i in range(1,5):
	for root,_,files in os.walk('./training_data/cropped/'+str(i)):
		for file in files:
			 postive_features.append(feature.hog(color.rgb2grey(io.imread(os.path.join(root,file)))));



# Filling the "negative_features" array with HOG fields of the negative sub-images from the uncropped images.


pos_of_hand = open('./training_data/bounding_boxes.csv').read().split('\n')[1:];
for k in range(300):
	prop = pos_of_hand[k].split(',')
	image = io.imread('./training_data/raw/' + prop[5] + '/' +prop[0]);


	y,x,_= image.shape

	initial_shift=30;
	for i in range(int(prop[1])+initial_shift,x-128,initial_shift):
		for j in range(int(prop[2])+initial_shift,y-128,initial_shift):
			# if (i,j)==(int(prop[1]),int(prop[2])): continue;
			negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128))));


	for i in range(min(int(prop[3])-initial_shift,x),128,-initial_shift):
		for j in range(min(int(prop[4])-initial_shift,y),128,-initial_shift):
			if (i,j)==(int(prop[3]),int(prop[4])): continue;
			negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i-128,i,j-128,j))));


negative_features=negative_features[:800]

X = postive_features + negative_features;
print len(postive_features),len(negative_features)
Y = [1]*len(postive_features) + [0]*len(negative_features);
svm.fit(X,Y);
# rf.fit(X,Y);
# mlp.fit(X,Y);
# etc.fit(X,Y);


#Dump the trained classifiers into into the dump folder
# with open('./dumps/mlp','wb') as d:
	# import pickle
	# pickle.dump(svm,d);
	# pickle.dump(rf,d);
	# pickle.dump(mlp,d);
	# pickle.dump(svm,d);


#TESTING---------------------------------------------------------------------------

positive_prediction=[]
negative_prediction=[]


# testing negative samples
for k in range(501,1000):
	prop = pos_of_hand[k].split(',')
	image = io.imread('./training_data/raw/' + prop[5] + '/' +prop[0]);



	y,x,_= image.shape

	initial_shift=30;
	for i in range(int(prop[1])+initial_shift,x-128,initial_shift):
		for j in range(int(prop[2])+initial_shift,y-128,initial_shift):
			if (i,j)==(int(prop[1]),int(prop[2])): continue;
			negative_prediction.append(feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128))));



	for i in range(min(int(prop[3]),x),128,-initial_shift):
		for j in range(min(int(prop[4]) ,y),128,-initial_shift):
			if (i,j)==(int(prop[3]),int(prop[4])): continue;
			negative_prediction.append(feature.hog(color.rgb2grey(crop_image(image,i-128,i,j-128,j))));


negative_prediction=negative_prediction[:200]



percentage=0;
for root,_,files in os.walk('./training_data/cropped/5'):
	files = files[::];
	for file in files:
		positive_prediction.append(feature.hog(color.rgb2grey(io.imread(os.path.join(root,file)))));


negative_results = svm.predict(negative_prediction);
positive_results = svm.predict(positive_prediction);

# positive_results = rf.predict(positive_prediction);
# negative_results = rf.predict(negative_prediction);
# #
# positive_results = mlp.predict(positive_prediction);
# negative_results = mlp.predict(negative_prediction[:len(positive_prediction)]);
#
# positive_results = etc.predict(positive_prediction);
# negative_results = etc.predict(negative_prediction);

print len(positive_prediction),len(negative_prediction)
print positive_results
print negative_results

percentage = float((sum(positive_results) + len(negative_results)-sum(negative_results)))/(len(positive_results)+len(negative_results));
print percentage;
