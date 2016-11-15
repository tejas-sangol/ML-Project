from skimage import io,transform,feature,color,data
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import os

def crop_image(arr,x,y,size):
    return arr[y:y+size,x:x+size];

# img=Image.open('./training_data/raw/5/1_35_1_cam1_0_raw.jpg')
# img=io.imread(img)
loaded_model = pickle.load(open('./dumps/svm','rb'))


new_width=128
new_height=128




min_box_size=100
max_box_size=245
box_size_step=10
box_pos_step=10


# tmp=img.load()
# print type(img)
# print np.shape(img)
# print img
# k=0

# im=crop_image(img,80,10,100)
# io.imsave("./sample_image.jpg",im)
def run(arg):
    root,dir=arg;
    
    if dir[0]=='.' : return;
    with open('./dataset/'+dir+'/'+dir+'_loc.csv') as file_list:
        file_list = file_list.read().split('\n')[1:];

        for entry in file_list:
            if len(entry)==0:continue;
            symbol = entry[entry.find('/')+1:entry.find('/')+3];
            # cordinates = map(int,entry.split(',')[1:]);

            img = io.imread('./dataset/'+ entry.split(',')[0]);

            image_width=np.shape(img)[1]
            image_height=np.shape(img)[0]
            print image_width,image_height

            max_image_x=0
            max_image_y=0
            max_image_box_size=0
            max_image_prob=0.0
            max_image=img

            for i in xrange(min_box_size,max_box_size,box_size_step):
                print i
                for y in xrange(0,image_height-i,box_pos_step):
                    for x in xrange(0,image_width-i,box_pos_step):
                        # print c,r,i
                        # im=img.crop((c,r,c+i,r+i))
                        im=crop_image(img,x,y,i)
                        # im=im.resize((new_width,new_height),Image.ANTIALIAS)
                        im=transform.resize(im,(new_height,new_width))
                        p=loaded_model.predict_proba([feature.hog(color.rgb2grey(im))])[0]
                        if(p[1] > max_image_prob):
                            max_image_prob=p[1]
                            max_image_x=x
                            max_image_y=y
                            max_image_box_size=i
                            max_image=im
            print i,":",max_image_prob,max_image_x,max_image_y,max_image_box_size
        # io.imsave("./test/"+str(i)+".jpg",max_image)
                # k+=1
                # if(k%10==0):im.save('./test/'+str(k)+'.jpg')
    # print max_image_x,max_image_y,(max_image_x+max_image_box_size),(max_image_y+max_image_box_size)
    # res=img.crop((max_image_x,max_image_y,(max_image_x+max_image_box_size),(max_image_y+max_image_box_size)))
    # res=res.resize((new_width,new_height),Image.ANTIALIAS)
    # im.save('./test/res.jpg')
            io.imsave('./test/' + entry.split(',')[0],max_image)

from multiprocessing import Pool;
p=Pool(4);
args=[];
for root,dirs,_ in os.walk('dataset'):
    for dir in dirs: args.append((root,dir));
p.map(run,args);
