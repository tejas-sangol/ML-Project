

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from skimage import io,transform,feature,color,data
import numpy as np
# from sliding_window import sliding_window
import os

def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
    return arr[start_coloumn:end_coloumn,start_row:end_row];



svm = svm.SVC();              #Untrained SVM Classifier
rf = RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=100,verbose=5);      #Untrained Random Forest Classifier
mlp = MLPClassifier();              #Untrained Neural network Classifier
etc = ExtraTreesClassifier();       #Untrained ExtraTrees Classifier




Y =[];
X=[];
for root,directories,_ in os.walk('dataset'):
    for dir in directories[:]:
        if dir[0]=='.' : continue;
        with open('./dataset/'+dir+'/'+dir+'_loc.csv') as file_list:
            file_list = file_list.read().split('\n')[1:];

            for entry in file_list:
                if len(entry)==0:continue;
                symbol = entry[entry.find('/')+1:entry.find('/')+3];
                cordinates = map(int,entry.split(',')[1:]);
                x1,y1,x2,y2 = cordinates[0],cordinates[1],cordinates[2],cordinates[3]

                image = io.imread('./dataset/'+ entry.split(',')[0]);

                X.append(feature.hog(color.rgb2grey(transform.resize(image,(120,160))),cells_per_block=(2,2)));
                Y.append([ord(symbol[0])-65])
            print dir
#
# grid = GridSearchCV(svm,paramaters,n-jobs=-1)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0)

rf.fit(train_x,train_y);
print rf.score(test_x, test_y);

with open('./multiclass_rf','wb') as d:
    import pickle
    # pickle.dump(svm,d);
    pickle.dump(rf,d);
    # pickle.dump(mlp,d);
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
