from skimage import transform


def sliding_window(width,height,size_of_window=128,alter=30):
	i=0;
	j=0;
	for i in range(0,width-size_of_window+1,alter):
		for j in range(0,height-size_of_window+1,alter):
			yield (i,j,i+size_of_window,j+size_of_window);
