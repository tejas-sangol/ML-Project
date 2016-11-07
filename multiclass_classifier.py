

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from skimage import io,transform,feature,color,data
import numpy as np
# from sliding_window import sliding_window
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
        files = files
        for file in files:
             features.append(feature.hog(color.rgb2grey(io.imread(os.path.join(root,file)))));
             Classes.append(i);


#
# grid = GridSearchCV(svm,paramaters,n-jobs=-1)
mlp.fit(features,Classes);


with open('./multiclass_mlp','wb') as d:
	import pickle
	# pickle.dump(svm,d);
	# pickle.dump(rf,d);
	pickle.dump(mlp,d);
	# pickle.dump(etc,d);

# features_test=[];
# classes_test=[]
#
# for i in range(1,6):
#     for root,_,files in os.walk('./training_data/cropped/'+str(i)):
#         files = files[161:]
#         for file in files:
#              features_test.append(feature.hog(color.rgb2grey(io.imread(os.path.join(root,file)))));
#              classes_test.append(i);
# print etc.score(features_test,classes_test);
