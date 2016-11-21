from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import skimage as ski
from skimage import io,transform,feature,color,data
import numpy as np
import pickle;
from multiprocessing import Pool;
import time
# from sliding_window import sliding_window
import os

def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];

def IOU(predicted,actual):
	cord1 = [max(actual[0],predicted[0]),max(actual[1],predicted[1])]
	cord2 = [min(actual[0]+actual[2],predicted[0]+predicted[2]),min(actual[1]+actual[2],predicted[1]+predicted[2])]

	intersectionArea = (cord2[0]-cord1[0])*(cord2[1]-cord1[1])
	unionArea = actual[2]*actual[2] + predicted[2]*predicted[2] - intersectionArea

	return float(intersectionArea)/unionArea



svm = svm.SVC(kernel='linear',C=10, probability=True);		        #Untrained SVM Classifier
rf = RandomForestClassifier(n_estimators = 40,
							random_state = 0,
							n_jobs = -1,
							verbose = 5);      #Untrained Random Forest Classifier
mlp = MLPClassifier();              #Untrained Neural network Classifier
etc = ExtraTreesClassifier();       #Untrained ExtraTrees Classifier


X,Y=[],[];


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

				if int(symbol[1])>=8: continue;
				image = io.imread('./dataset/'+ entry.split(',')[0]);

				shift=40;
				i=0;
				height,width,_=image.shape
				for x in xrange(0,width-128,shift):
					for y in xrange(0,height-128,shift):
						percentage_hand = IOU([x,y,128],[x1,y1,x2-x1]);

						# time.sleep(0.1)
						i+=1;
						if percentage_hand >=0.9:
							X.append(feature.hog(color.rgb2grey(crop_image(image,x,x+128,y,y+128)),cells_per_block=(2,2)));
							Y+=[1];
						elif percentage_hand<0.5:
							X.append(feature.hog(color.rgb2grey(crop_image(image,x,x+128,y,y+128)),cells_per_block=(2,2)));
							Y+=[0];
				print len(X);
				# time.sleep(0.1)

# print np.array(X).shape
# svm.fit(X,Y);

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0)

rf.fit(train_x, train_y)
print rf.score(test_x, test_y);

# mlp.fit(X,Y);
# etc.fit(X,Y);


# Dump the trained classifiers into into the dump folder
# with open('./dumps/svm','wb') as d:
# 	pickle.dump(svm,d);

with open('./dumps/rf','wb') as d:
	pickle.dump(rf,d);


# with open('./dumps/mlp','wb') as d:
# 	pickle.dump(mlp,d);


# with open('./dumps/etc','wb') as d:
# 	pickle.dump(etc,d);


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
