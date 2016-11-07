from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import skimage as ski
from skimage import io,transform,feature,color,data
import numpy as np
# from sliding_window import sliding_window
import os

def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];



svm = svm.SVC(kernel='linear',C=1, probability=True);		        #Untrained SVM Classifier
rf = RandomForestClassifier();      #Untrained Random Forest Classifier
mlp = MLPClassifier();              #Untrained Neural network Classifier
etc = ExtraTreesClassifier();       #Untrained ExtraTrees Classifier



postive_features =[]    # Will be filled with the HOG descriptors of the cropped images(128 x 128 pixels).
negative_features =[];  # Will be filled with hard negatives from the uncropped images.

#Filling the "postive_features" array with HOG fields.
# Methods used ---
# 1. io.imread() --> Loads the image as a 3D array from the file
# 2. color.rgb2grey --> Normalizes the 3D coloured array to 2D greyscaled array necessary for HOG.
# 3. features.hog --> Ruturns the HOG Descriptor for the 2D array
for i in range(1,6):
	for root,_,files in os.walk('./training_data/cropped/'+str(i)):
		for file in files:
			 postive_features.append(feature.hog(color.rgb2grey(io.imread(os.path.join(root,file)))));



# Filling the "negative_features" array with HOG fields of the negative sub-images from the uncropped images.


pos_of_hand = open('./training_data/bounding_boxes.csv').read().split('\n')[1:];
for k in range(1000):
	prop = pos_of_hand[k].split(',')
	image = io.imread('./training_data/raw/' + prop[5] + '/' +prop[0]);


	y,x,_= image.shape

	initial_shift=40;
	for i in range(int(prop[1])+initial_shift,x-128,initial_shift):
		for j in range(int(prop[2])+initial_shift,y-128,initial_shift):
			# if (i,j)==(int(prop[1]),int(prop[2])): continue;
			negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128))));


	for i in range(min(int(prop[3])-initial_shift,x),128,-initial_shift):
		for j in range(min(int(prop[4])-initial_shift,y),128,-initial_shift):
			if (i,j)==(int(prop[3]),int(prop[4])): continue;
			negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i-128,i,j-128,j))));

	if len(negative_features) >=2000: break;



X = postive_features + negative_features;
print len(postive_features),len(negative_features)
Y = [1]*len(postive_features) + [0]*len(negative_features);
svm.fit(X,Y);
# rf.fit(X,Y);
mlp.fit(X,Y);
# etc.fit(X,Y);


# Dump the trained classifiers into into the dump folder
with open('./dumps/svm','wb') as d:
	import pickle
	pickle.dump(svm,d);
	# pickle.dump(rf,d);
	pickle.dump(mlp,d);
	# pickle.dump(etc,d);


#TESTING---------------------------------------------------------------------------

# with open('./dumps/svm','rb') as f:
# 	import pickle;
# 	svm = pickle.load(f);
#
# for i in range(1,11):
# 	entry = open('./training_data/bounding_boxes.csv').read().split('\n')[i].split(',');
# 	name = entry[0];
# 	image = io.imread('./training_data/raw/1/'+name)
# 	test_features =[]
#
# 	maximum_confidence=0;
# 	avg_x=0;
# 	avg_y=0;
# 	count_max=0;
# 	for window in sliding_window(offset=10):
# 		x1,y1,x2,y2 = window;
#
# 		cropped_image = crop_image(image,x1,x2,y1,y2);
# 		tmp=map(lambda x:round(x,6),svm.predict_proba([feature.hog(color.rgb2grey(cropped_image))])[0])
# 		if tmp[1] > maximum_confidence :
# 			maximum_confidence=tmp[1]
# 			avg_x=x1
# 			avg_y=y1
# 			count_max=1;
# 		elif tmp[1] == maximum_confidence:
# 			avg_x += x1;
# 			avg_y += y1;
# 			count_max += 1;
#
# 		print window,tmp
# 	print maximum_confidence,float(avg_x)/count_max,float(avg_y)/count_max;
