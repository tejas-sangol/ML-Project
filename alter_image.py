from skimage import io

def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];

pos_of_hand = open('./training_data/bounding_boxes.csv').read().split('\n')[1:];

prop = pos_of_hand[0].split(',')

image = io.imread('./training_data/raw/1/'+prop[0]);

io.imsave('test.jpg',crop_image(image,int(prop[1])+90,int(prop[3])+40,int(prop[2]),int(prop[4])))
