from skimage import io,transform,feature,color,data
import numpy as np
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
from skimage.transform import pyramid_gaussian,pyramid_expand
def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];
# string.decode('utf-8')
loaded_model = pickle.load(open('./dumps/mlp','rb'))
pred = [0,0,0,0]
max_image = None;
# iterates over first 10 images
def run(i):
	j_value =i;
	entry = open('./training_data/bounding_boxes.csv').read().split('\n')[i].split(',')
	name = entry[0];
	image = io.imread('./training_data/raw/3/'+name)
	pred = [0,0,0,0]
	#downscaled images
	downscale_pyramid = pyramid_gaussian(image,downscale = 1.1) # generator

	#upscaled images
	upscale_pyramid = []
	im = image
	for j in range(0,10):
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

	with open('./multiclass_mlp','rb') as f:
		import pickle;
		m_svm = pickle.load(f);

	print m_svm.predict([max_image]);
	io.imsave('check'+str(j_value)+'.jpg',cropped_image);
	entry[1:] = map(int,entry[1:]);

	#Intersection over Union error calculation
	actual = [entry[1],entry[2],entry[3]-entry[1]] # [x,y,size]
	# pred = pred[:3]

	cord1 = [max(actual[0],pred[0]),max(actual[1],pred[1])]
	cord2 = [min(actual[0]+actual[2],pred[0]+pred[2]),min(actual[1]+actual[2],pred[1]+pred[2])]

	intersectionArea = (cord2[0]-cord1[0])*(cord2[1]-cord1[1])
	unionArea = actual[2]*actual[2] + pred[2]*pred[2] - intersectionArea

	error = float(intersectionArea)/unionArea

	print (pred,actual,error)

p=Pool(4);
p.map(run,[i for i in range(401,411)]);
# run(611);
