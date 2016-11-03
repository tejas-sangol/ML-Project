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
#
#
#
# svm = svm.SVC(kernel='linear',probability=True,C=100);              #Untrained SVM Classifier
# rf = RandomForestClassifier();      #Untrained Random Forest Classifier
# mlp = MLPClassifier();              #Untrained Neural network Classifier
# etc = ExtraTreesClassifier();       #Untrained ExtraTrees Classifier
#
#
#
#
# Classes =[];
# features=[];
# for i in range(1,6):
# 	for root,_,files in os.walk('./training_data/cropped/'+str(i)):
# 		# files = files[:120]
# 		for file in files:
# 			 features.append(feature.hog(color.rgb2grey(io.imread(os.path.join(root,file)))));
# 			 Classes.append(i);
#
#
#
#
# X = features
# Y = Classes
# svm.fit(X,Y);
# # rf.fit(X,Y);
# # mlp.fit(X,Y);
# # etc.fit(X,Y);
#
#
#
# with open("multiclass_svm",'wb') as f:
# 	import pickle
# 	pickle.dump(svm,f)
#
#




# LOAD PICKLED OBJECT----------------------
import pickle
with open('multiclass_svm','rb') as f:
	svm = pickle.load(f)
# --------------------------------

# estimate=[0 for _ in xrange(6)]
image = io.imread('./training_data/raw/5/17_35_1_cam1_0_raw.jpg');
x=101;
y=18;
# for i in sliding_window(offset=10):
# 	x1,y1,x2,y2 = i;
# 	ans = svm.predict_proba([feature.hog(color.rgb2grey(crop_image(image,x1,x2,y1,y2)))])[0]
# 	estimate = [max(estimate[j],ans[j]) for j in xrange(6)]
# 	print i, map(lambda x:round(x,6),ans);
print svm.predict_proba([feature.hog(color.rgb2grey(crop_image(image,x,x+128,y,y+128)))]);
io.imsave('trail.jpg',(crop_image(image,x,x+128,y,y+128)))
# print estimate
