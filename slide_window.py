from skimage import io,transform,feature,color,data
import numpy as np
import pickle

from skimage.transform import pyramid_gaussian,pyramid_expand

loaded_model = pickle.load(open('./dumps/svm','rb'))

# iterates over first 10 images
def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];


def generate_downscaled_images(image,downscale=1.1):
	return pyramid_gaussian(image,downscale);

def generate_upscaled_images(image,upscale=1.1,count=10):
	for i in range(count):
		image = pyramid_expand(image,upscale);
		yield image;

def sliding_window(image,shift=10):
	predicted = [0]*4;
	# max_image =None;

	a = 1
	for im in generate_downscaled_images(image,1.1):
		height,width,_= im.shape
		if width < 128 or height < 128 : continue;
		for i in range (0,width-128,shift):
			for j in range (0,height -128,shift):
				w = loaded_model.predict_proba([feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128)))])[0]
				if (predicted[3] < w[1]):
					predicted = [i*a,j*a,128*a,w[1]]
					# max_image=image;
		a = a*1.1;

	a = 1/1.1
	for im in generate_upscaled_images(image,1.1,10):
		height,width,_= im.shape
		for i in range (0,width-128,shift):
			for j in range (0,height -128,shift):
				if i+128<=width or j+128 <=height : continue;
				w = loaded_model.predict_proba([feature.hog(color.rgb2grey(crop_image(image,i,i+128,j,j+128)))])[0]
				if (predicted[3] < w[1]):
					predicted = [i*a,j*a,128*a,w[1]]
					# max_image=image;
		a = a/1.1;


		# return (predicted,max_image);
		return predicted;


def IOU(predicted,actual):
	cord1 = [max(actual[0],predicted[0]),max(actual[1],predicted[1])]
	cord2 = [min(actual[0]+actual[2],predicted[0]+predicted[2]),min(actual[1]+actual[2],predicted[1]+predicted[2])]

	intersectionArea = (cord2[0]-cord1[0])*(cord2[1]-cord1[1])
	unionArea = actual[2]*actual[2] + predicted[2]*predicted[2] - intersectionArea

	return float(intersectionArea)/unionArea



for i in range(1,11):
	entry = open('./training_data/bounding_boxes.csv').read().split('\n')[i].split(',');
	name = entry[0];
	image = io.imread('./training_data/raw/1/'+name)

	predicted = sliding_window(image,shift=10);    #predicted = [x-cordinate,y-cordinate,size of the bounding box]

	entry[1:] = map(int,entry[1:]);

	actual = [entry[1],entry[2],entry[3]-entry[1]] # [x,y,size]


	error = IOU(predicted,actual);

	print (error)
