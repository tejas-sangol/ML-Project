from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import skimage as ski
from skimage import io,transform,feature,color,data
from multiprocessing import Pool,Process,Lock
import numpy as np
import math

l1=['user_3','user_4','user_5','user_6']
l2=['user_7','user_9','user_10','user_11']
l3=['user_12','user_13','user_14','user_15']
l4=['user_16','user_17','user_18','user_19']


lock=Lock()

img_height=240.0
img_width=320.0
# img_size=80
# new_img_height=48
# new_img_width=64
# step=40

def crop_image(arr,x1,y1,x2,y2):
    return arr[y1:y2,x1:x2];

def get_index(s):
    return int(s.split('_')[1])

def get_images(l,new_img_width,new_img_height,step):
    # lock.acquire()
    for i in l:
    # i=l
        f=open('./dataset/'+i+'/'+i+'_loc.csv')
        bounding_boxes=f.read().split('\n')[1:-1]
        # reg_X[i]=[]
        # reg_Y[i]=[]
        for j in bounding_boxes:
            im,x1,y1,x2,y2=j.split(',')
            x1=int(x1);y1=int(y1);x2=int(x2);y2=int(y2)
            size=x2-x1  #Assuming bounding box is a square

            img=io.imread('./dataset/'+im)
            # img=color.rgb2grey(img)
            red=img[:,:,0]
            grn=img[:,:,1]
            blu=img[:,:,2]
            # img=transform.resize(img,(new_img_height,new_img_width))

            reg_X[get_index(i)].append(feature.hog(img))
                    # reg_X[get_index(i)].append(feature.hog(cr_img,cells_per_block=(2,2)))
            p1=float(x1)/img_width;
            q1=float(y1)/img_height;
            p2=float(x2)/img_width;
            q2=float(y2)/img_height;

            reg_Y[get_index(i)].append([p1,q1,p2,q2])
                    # io.imsave('./sample/neg/'+i+'/'+str(c)+".jpg",cr_img)
                    # io.imsave('./sample/neg/'+i+'/'+str(c)+"_"+str(p)+".jpg",cr_img)
                    
            print im
    # lock.release()

# lreg = linear_model.Ridge(alpha=0.5)                      
# lreg = linear_model.LinearRegression(n_jobs=-1)   
lreg=MLPRegressor()           


def train(l):
    X=[]
    Y=[]
    for i in l:
        for j in xrange(len(reg_X[(get_index(i))])):
            X.append(reg_X[get_index(i)][j])
            Y.append(reg_Y[get_index(i)][j])
    lreg.fit(X,Y)
 


def test(l):
    X=[]
    Y=[]
    for i in l:
        for j in xrange(len(reg_X[(get_index(i))])):
            X.append(reg_X[get_index(i)][j])
            Y.append(reg_Y[get_index(i)][j])
    return lreg.score(X,Y)




def search(new_img_width,new_img_height,step):
    get_images(l,new_img_width,new_img_height,step)
    # train(['user_3','user_4','user_5','user_7','user_9','user_10','user_12','user_13','user_14','user_16','user_17','user_18'])
    # print new_img_width,new_img_height,step,test(['user_6','user_11','user_15','user_19'])
    train(['user_3','user_4','user_5'])
    print new_img_width,new_img_height,step,test(['user_6'])


# get_images(l1+l2+l3+l4)
# l=l1+l2+l3+l4
l=l1

# for i in xrange(8,30):
#     for j in xrange(20,80):
#         reg_X=[[] for _ in xrange(20)]
#         reg_Y=[[] for _ in xrange(20)]
#         search(i*4,i*3,j)


reg_X=[[] for _ in xrange(20)]
reg_Y=[[] for _ in xrange(20)]
search(img_width,img_height,10)


# p1=Process(target=get_images,args=(lock,['user_3'],));
# p1.start();
# p2=Process(target=get_images,args=(lock,['user_4'],));
# p2.start();
# p3=Process(target=get_images,args=(lock,['user_5'],));
# p3.start();
# p4=Process(target=get_images,args=(lock,['user_6'],));
# p4.start();
# p1.join();
# p2.join();
# p3.join();
# p4.join();
# p=Pool(4)
# p.map(get_images,l)
# train(['user_3','user_4','user_5','user_7','user_9','user_10','user_12','user_13','user_14','user_16','user_17','user_18'])
# get_images(l)
# train(['user_3','user_4','user_5','user_7','user_9','user_10','user_12','user_13','user_14','user_16','user_17','user_18'])
# print test(['user_6','user_11','user_15','user_19'])


# get_images(l)
# train(['user_3','user_4','user_5'])
# print test(['user_6'])


# get_images(l1+l2)
# train(['user_3','user_4','user_5','user_7','user_9','user_10'])
# print test(['user_6','user_11'])


# l=l1+l2+l3+l4
# l=["user_3","user_4"]
# l=l1
# get_images(["user_3","user_4","user_5","user_6"])
# print len(reg_X[get_index("user_3")])
# print len(reg_Y[get_index("user_3")])
# print len(reg_X[get_index("user_4")])
# print len(reg_Y[get_index("user_4")])
# print len(reg_Y[get_index("user_5")])
# print len(reg_Y[get_index("user_5")])
# print len(reg_Y[get_index("user_6")])
# print len(reg_Y[get_index("user_6")])

# lreg.fit(reg_X[get_index("user_3")] + reg_X[get_index("user_4")] + reg_X[get_index("user_5")] ,reg_Y[get_index("user_3")] + reg_Y[get_index("user_4")] + reg_Y[get_index("user_5")] )
# print lreg.score(reg_X[get_index("user_6")],reg_Y[get_index("user_6")])



# for key in reg_Y.keys():
    # print "hello",key

# train(["user_3"])
# print test(["user_4"])

# train(['user_3','user_4','user_5'])
# s1=test(['user_6'])
# print s1

# train(l2+l3+l4)
# s1=test(l1)
# print s1

# train(l1+l3+l4)
# s2=test(l2)
# print s2

# train(l1+l2+l4)
# s3=test(l3)
# print s3

# train(l1+l2+l3)
# s4=test(l4)
# print s4

# print float(s1+s2+s3+s4)/4.0
