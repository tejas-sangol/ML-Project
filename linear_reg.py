from sklearn import linear_model
import skimage as ski
from skimage import io,transform,feature,color,data
from multiprocessing import Pool
import numpy as np

l1=['user_3','user_4','user_5','user_6']
l2=['user_7','user_9','user_10','user_11']
l3=['user_12','user_13','user_14','user_15']
l4=['user_16','user_17','user_18','user_19']

reg_X={}
reg_Y={}

img_height=240
img_width=320
img_size=80

def crop_image(arr,x1,y1,x2,y2):
    return arr[y1:y2,x1:x2];

def IOU(predicted,actual):
	cord1 = [max(actual[0],predicted[0]),max(actual[1],predicted[1])]
	cord2 = [min(actual[0]+actual[2],predicted[0]+predicted[2]),min(actual[1]+actual[2],predicted[1]+predicted[2])]
	intersectionArea = (cord2[0]-cord1[0])*(cord2[1]-cord1[1])
	unionArea = actual[2]*actual[2] + predicted[2]*predicted[2] - intersectionArea
	return abs(float(intersectionArea)/unionArea)

def get_images(l):
    for i in l:
    # i=l
        f=open('./dataset/'+i+'/'+i+'_loc.csv')
        bounding_boxes=f.read().split('\n')[1:-1]
        reg_X[i]=[]
        reg_Y[i]=[]
        for j in bounding_boxes:
            c=0
            im,x1,y1,x2,y2=j.split(',')
            x1=int(x1);y1=int(y1);x2=int(x2);y2=int(y2)
            img=io.imread('./dataset/'+im)
            img=color.rgb2grey(img)
            size=x2-x1  #Assuming bounding box is a square
            for y in xrange(0,img_height-size,100):
                for x in xrange(0,img_width-size,100):
                    p=IOU([x,y,size],[x1,y1,size])
                    cr_img=crop_image(img,int(x),int(y),int(x+size),int(y+size))
                    cr_img=transform.resize(cr_img,(img_height,img_width))
                    # feat=feature.hog(cr_img)    	            
                    reg_X[i].append(feature.hog(cr_img))
                    reg_Y[i].append(p)
                    # io.imsave('./sample/neg/'+i+'/'+str(c)+".jpg",cr_img)
                    # io.imsave('./sample/neg/'+i+'/'+str(c)+"_"+str(p)+".jpg",cr_img)
                    c+=1
            print im,c

lreg = linear_model.LinearRegression(n_jobs=-1)		        #Untrained SVM Classifier


# def train(l):
#     X=[]
#     Y=[]
#     for i in l:
#         for j in xrange(len(reg_Y[i])):
#             X.append(feature.hog(reg_X[i][j]))
#             Y.append(reg_Y[i][j])
#     lreg.fit(X,Y)

# def test(l):
#     X=[]
#     Y=[]
#     for i in l:
#         for j in xrange(len(reg_Y[i])):
#             X.append(feature.hog(reg_X[i][j]))
#             Y.append(reg_Y[i][j])
#     return lreg.score(X,Y)



# l=l1+l2+l3+l4
# l=["user_3","user_4"]
# l=l1
get_images(["user_3","user_4"])
print len(reg_X["user_3"])
print len(reg_Y["user_3"])
print len(reg_X["user_4"])
print len(reg_Y["user_4"])
lreg.fit(reg_X["user_3"],reg_Y["user_3"])
print lreg.score(reg_X["user_4"],reg_Y["user_4"])
# p=Pool(4)
# p.map(get_images,l)

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
