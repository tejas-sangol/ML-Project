from skimage import io,transform,feature,color,data
import numpy as np
import matplotlib.pyplot as plt
import pickle,os;
from multiprocessing import Pool
from skimage.transform import pyramid_gaussian,pyramid_expand
import sys;
def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];
# string.decode('utf-8')
loaded_model = pickle.load(open('./dumps/svm','rb'));
# with open('./multiclass_mlp','rb') as f:
# 	import pickle;
# 	m_svm = pickle.load(f);

# iterates over first 10 images
def run(arg):
	pred=[0,0,0,0];
	symbol,dir,entry = arg;
	image = io.imread('./dataset/'+ entry.split(',')[0]);

	#downscaled images
	downscale_pyramid = pyramid_gaussian(image,downscale = 1.1) # generator

	#upscaled images
	upscale_pyramid = []
	im = image
	for j in range(0,5):
		im = pyramid_expand(im,upscale = 1.1)
		upscale_pyramid.append(im)


	shift=5;
	a = 1
	for im in downscale_pyramid:
		height,width,_= im.shape
		if width < 128 or height < 128 : continue;
		for i in range (0,width-128,shift):
			for j in range (0,height -128,shift):
				w = loaded_model.predict_proba([feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128)))])[0];
				if (pred[3] < w[1]):
					pred = [i*a,j*a,128*a,w[1]];
					max_image = feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128)));
					cropped_image=crop_image(image,i,i+128,j,j+128);
		a = a*1.1

	a = 1/1.1
	for im in upscale_pyramid:
		height,width,_= im.shape
		if width < 128 or height < 128 : continue;
		for i in range (0,width-128,shift):
			for j in range (0,height -128,shift):
				if i+128<=width or j+128 <=height : continue;
				w = loaded_model.predict_proba([feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128)))])[0];
				if (pred[3] < w[1]):
					pred = [i*a,j*a,128*a,w[1]];
					max_image = feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128)));
					cropped_image=crop_image(image,i,i+128,j,j+128);
		a = a/1.1

	io.imsave('./test/'+str(dir)+'/'+symbol+'.jpg',cropped_image);

	entry = map(int,entry.split(',')[1:]);
	#
	# #Intersection over Union error calculation
	actual = [entry[0],entry[1],entry[2]-entry[0]] # [x,y,size]
	# # pred = pred[:3]
	#
	# cord1 = [max(actual[0],pred[0]),max(actual[1],pred[1])]
	# cord2 = [min(actual[0]+actual[2],pred[0]+pred[2]),min(actual[1]+actual[2],pred[1]+pred[2])]
	#
	# intersectionArea = (cord2[0]-cord1[0])*(cord2[1]-cord1[1])
	# unionArea = actual[2]*actual[2] + pred[2]*pred[2] - intersectionArea
	#
	# iou  = float(intersectionArea)/unionArea;
	#

	# prediction = m_svm.predict_proba([max_image])[0];
	# pred_max = max(prediction);
	# pred_index = [i for i in range(len(prediction)) if prediction[i]==pred_max ][0] + 1;

	print dir+'/'+symbol,pred,actual,symbol[0]
	sys.stdout.flush();

p=Pool(4);

args=[];
for root,directories,_ in os.walk('dataset'):
	for dir in directories:
		if dir[0]=='.' : continue;
		with open('./dataset/'+dir+'/'+dir+'_loc.csv') as file_list:
			file_list = file_list.read().split('\n')[1:];

			for entry in file_list:
				if len(entry)==0:continue;
				symbol = entry[entry.find('/')+1:entry.find('/')+3];
				if int(symbol[1])<8: continue;
				args.append((symbol,dir,entry));
p.map(run,args);
# run(('A8', 'user_10', 'user_10/A8.jpg,195,61,305,171'));
