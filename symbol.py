# This id for part 2 of the problem where one has to classify the cropped images into classes -> [1,2,3,4,5].
# Each classifier is trained with 120 images from each class and tested on the remaining 80.
# 'features' array holds the HOG descriptors of the cropped images and 'Classes' array holds the corresponding classes to train.
# Variables a,b are used to calculate the percentage of correct classification on the test set each time the script is run.

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import skimage as ski
from skimage import io,transform,feature,color,data
import numpy as np
from sliding_window import sliding_window
import os

def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];



svm = svm.SVC(kernel='linear',probability=True);              #Untrained SVM Classifier
rf = RandomForestClassifier();      #Untrained Random Forest Classifier
mlp = MLPClassifier();              #Untrained Neural network Classifier
etc = ExtraTreesClassifier();       #Untrained ExtraTrees Classifier




Classes =[];
features=[];
for i in range(1,6):
	for root,_,files in os.walk('./training_data/cropped/'+str(i)):
		files = files[:120]
		for file in files:
			 features.append(feature.hog(color.rgb2grey(io.imread(os.path.join(root,file)))));
			 Classes.append(i);



#Train negative samples
negative_features=[];

pos_of_hand = open('./training_data/bounding_boxes.csv').read().split('\n')[1:];
for k in range(1000):
	if k%200 > 120: continue;
	prop = pos_of_hand[k].split(',')
	image = io.imread('./training_data/raw/' + prop[5] + '/' +prop[0]);


	y,x,_= image.shape


	initial_shift=50;
	for i in range(int(prop[1])+initial_shift,x-128,initial_shift/5):
		for j in range(int(prop[2])+initial_shift,y-128,initial_shift/5):
			# if (i,j)==(int(prop[1]),int(prop[2])): continue;
			negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128))));


	for i in range(min(int(prop[3])-initial_shift,x),128,-initial_shift/5):
		for j in range(min(int(prop[4])-initial_shift,y),128,-initial_shift/5):
			if (i,j)==(int(prop[3]),int(prop[4])): continue;
			negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i-128,i,j-128,j))));

	if len(negative_features) >=4000: break;



X = features + negative_features;
Y = Classes + [0]*len(negative_features);
print len(X)-len(negative_features), len(negative_features)
svm.fit(X,Y);
# rf.fit(X,Y);
# mlp.fit(X,Y);
# etc.fit(X,Y);


#
# with open("multiclass_svm",'wb') as f:
# 	import pickle
# 	pickle.dump(svm,f)






# LOAD PICKLED OBJECT----------------------
# import pickle
# with open('multiclass_svm','rb') as f:
# 	svm = pickle.load(f)
# --------------------------------

estimate=[0 for _ in xrange(6)]
image = io.imread('./training_data/raw/4/7_34_1_cam1_4_raw.jpg');

for i in sliding_window(offset=10):
	x1,y1,x2,y2 = i;
	ans = svm.predict_proba([feature.hog(color.rgb2grey(crop_image(image,x1,x2,y1,y2)))])[0]
	estimate = [max(estimate[j],ans[j]) for j in xrange(6)]
	print i, map(lambda x:round(x,6),ans);
	# print i,svm.predict([feature.hog(color.rgb2grey(crop_image(image,x1,x2,y1,y2)))]);
print estimate
