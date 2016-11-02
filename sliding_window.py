from skimage import io
prop = open('./training_data/bounding_boxes.csv').read().split('\n')[-1].split(',');

image = io.imread('./training_data/raw/5/' + prop[0]);

def crop_image(arr,start_row,end_row,start_coloumn,end_coloumn):
	return arr[start_coloumn:end_coloumn,start_row:end_row];


x,y,_=image.shape
for i in range(x-128,10):
    for j in range(y-128,10):
        cropped_image = crop_image(image,i,i+128,j,j+128);
