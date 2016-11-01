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
import Image
import os

def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];



svm = svm.LinearSVC();              #Untrained SVM Classifier
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

# svm.fit(features,Classes);
# rf.fit(features,Classes);
# mlp.fit(features,Classes);
etc.fit(features,Classes);
a=0;
b=0;
for i in range(1,6):
    for root,_,files in os.walk('./training_data/cropped/'+str(i)):
        files = files[121:200]
        for file in files:
            # ans = svm.predict([feature.hog(color.rgb2grey(io.imread(os.path.join(root,file))))]);
            # ans = rf.predict([feature.hog(color.rgb2grey(io.imread(os.path.join(root,file))))]);
            # ans = mlp.predict([feature.hog(color.rgb2grey(io.imread(os.path.join(root,file))))]);
            ans = etc.predict([feature.hog(color.rgb2grey(io.imread(os.path.join(root,file))))]);
            print ans;
            if ans[0]==i: a+=1;
            b+=1;
print float(a)/b;
