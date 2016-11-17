from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import skimage as ski
from skimage import io,transform,feature,color,data
import numpy as np
from keras.utils import np_utils
import pickle;
from multiprocessing import Pool;
# from sliding_window import sliding_window
import os

def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];



svm = svm.SVC(kernel='linear',C=10, probability=True);		        #Untrained SVM Classifier
rf = RandomForestClassifier(n_estimators=100,random_state=100,n_jobs=-1,max_leaf_nodes=100);      #Untrained Random Forest Classifier
mlp = MLPClassifier();              #Untrained Neural network Classifier
etc = ExtraTreesClassifier();       #Untrained ExtraTrees Classifier
ada=AdaBoostClassifier(n_estimators=1000,random_state=100);


postive_features =[];    # Will be filled with the HOG descriptors of the cropped images(128 x 128 pixels).
negative_features =[];  # Will be filled with hard negatives from the uncropped images.

pca_train=[];
pca_class =[];
for root,directories,_ in os.walk('dataset'):
	for dir in directories[:4]:
		if dir[0]=='.' : continue;
		with open('./dataset/'+dir+'/'+dir+'_loc.csv') as file_list:
			file_list = file_list.read().split('\n')[1:];

			for entry in file_list:
				if len(entry)==0:continue;
				symbol = entry[entry.find('/')+1:entry.find('/')+3];
				cordinates = map(int,entry.split(',')[1:]);
				x1,y1,x2,y2 = cordinates[0],cordinates[1],cordinates[2],cordinates[3]

				# if int(symbol[1])>=8: continue;
				image = io.imread('./dataset/'+ entry.split(',')[0]);

		# 		cropped_image = transform.resize(crop_image(image,x1,x2,y1,y2),(128,128));
		#
		# 		postive_features.append(feature.hog(color.rgb2grey(cropped_image)));
		#
		# 		y,x,_= image.shape
		#
		# 		initial_shift=30;
		# 		shift =30;
		# 		temp_negative_features=[];
		# 		for i in range(x1+initial_shift,x-128,shift):
		# 			for j in range(y1+initial_shift,y-128,shift):
		# 				temp_negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128))));
		#
		#
		# 		for i in range(min(x2-initial_shift,x),128,-shift):
		# 			for j in range(min(y2-initial_shift,y),128,-shift):
		# 				temp_negative_features.append(feature.hog(color.rgb2grey(crop_image(image,i-128,i,j-128,j))));
		#
		# 		negative_features = negative_features + temp_negative_features;
		#
		#
		# print len(postive_features),len(negative_features),dir
				pca_train.append(feature.hog(color.rgb2grey(image)));
				pca_class.append(ord(symbol[0])-65);
		print dir;

# X = postive_features + negative_features;
# Y = [1]*len(postive_features) + [0]*len(negative_features);
pca = PCA()
pca.fit(pca_train);

X=pca.fit_transform(pca_train);
X_train=X[:400];
X_test=X[400:];
Y_train=pca_class[:400];
Y_test=pca_class[400:];

# for i in X: print type(i),len(i)
svm.fit(X_train,Y_train);
# rf.fit(X,Y);
# mlp.fit(X,Y);
# etc.fit(X,Y);
# ada.fit(X,Y);

print svm.score(X_test,Y_test);
# Dump the trained classifiers into into the dump folder
# with open('./dumps/svm2','wb') as d:
# 	pickle.dump(svm2,d);

# with open('./dumps/rf','wb') as d:
# 	pickle.dump(rf,d);


# with open('./dumps/mlp','wb') as d:
# 	pickle.dump(mlp,d);


# with open('./dumps/etc','wb') as d:
# 	pickle.dump(etc,d);

# with open('./dumps/ada','wb') as d:
# 	pickle.dump(ada,d);


#TESTING---------------------------------------------------------------------------
#
# with open('./dumps/svm','rb') as f:
# 	import pickle;
# 	svm = pickle.load(f);
# test_postitive = [];
# test_negative=[]
# for root,directories,_ in os.walk('dataset'):
# 	for dir in directories:
# 		if dir[0]=='.' : continue;
# 		with open('./dataset/'+dir+'/'+dir+'_loc.csv') as file_list:
# 			file_list = file_list.read().split('\n')[1:];

# 			for entry in file_list:
# 				if len(entry)==0:continue;
# 				symbol = entry[entry.find('/')+1:entry.find('/')+3];
# 				cordinates = map(int,entry.split(',')[1:]);
# 				x1,y1,x2,y2 = cordinates[0],cordinates[1],cordinates[2],cordinates[3]

# 				if int(symbol[1])<8: continue;
# 				image = io.imread('./dataset/'+ entry.split(',')[0]);

# 				cropped_image = transform.resize(crop_image(image,x1,x2,y1,y2),(128,128));

# 				test_postitive.append(feature.hog(color.rgb2grey(cropped_image)));

# 				y,x,_= image.shape

# 				initial_shift=40;
# 				shift =40;
# 				for i in range(x1+initial_shift,x-128,shift):
# 					for j in range(y1+initial_shift,y-128,shift):
# 						test_negative.append(feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128))));


# 				for i in range(min(x2-initial_shift,x),128,-shift):
# 					for j in range(min(y2-initial_shift,y),128,-shift):
# 						test_negative.append(feature.hog(color.rgb2grey(crop_image(image,i-128,i,j-128,j))));

# 				if len(test_negative) >=2000: continue;
# test_negative = test_negative[:len(test_postitive)]
# X_test = test_postitive + test_negative;
# print len(test_postitive),len(test_negative);
# Y_test = [1]*len(test_postitive) + [0]*len(test_negative);

# print svm.score(X_test,Y_test);

	# maximum_confidence=0;
	# avg_x=0;
	# avg_y=0;
	# count_max=0;
	# for window in sliding_window(offset=10):
	# 	x1,y1,x2,y2 = window;
	#
	# 	cropped_image = crop_image(image,x1,x2,y1,y2);
	# 	tmp=map(lambda x:round(x,6),svm.predict_proba([feature.hog(color.rgb2grey(cropped_image))])[0])
	# 	if tmp[1] > maximum_confidence :
	# 		maximum_confidence=tmp[1]
	# 		avg_x=x1
	# 		avg_y=y1
	# 		count_max=1;
	# 	elif tmp[1] == maximum_confidence:
	# 		avg_x += x1;
	# 		avg_y += y1;
	# 		count_max += 1;
	#
	# 	print window,tmp
	# print maximum_confidence,float(avg_x)/count_max,float(avg_y)/count_max;
	# print svm.predict_proba([feature.hog(color.rgb2grey(image))])[0]
