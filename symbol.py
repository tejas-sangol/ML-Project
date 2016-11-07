

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from skimage import io,transform,feature,color,data
import numpy as np
from sliding_window import sliding_window
import os

def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];



svm = svm.SVC();              #Untrained SVM Classifier
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




# X = features
# Y = Classes
# parameters = {
# 'kernel' :['linear','rbf','poly'],
# 'C':[.001,.01,.1,1,10,100,1000],
# 'gamma':[.001,.01,.1,1,10,100,1000]
# }
#
# classifier = GridSearchCV(svm,parameters,n_jobs=-1);
#
# print classifier.fit(X,Y);
