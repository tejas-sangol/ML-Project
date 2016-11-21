from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import skimage as ski
from skimage import io,transform,feature,color,data
import numpy as np

l1=['user_3','user_4','user_5','user_6']
l2=['user_7','user_9','user_10','user_11']
l3=['user_12','user_13','user_14','user_15']
l4=['user_16','user_17','user_18','user_19']

pos=[[]]*100;
neg=[[]]*100;

def get_position(string):
    return int(string.split('_')[1]);


img_height=240
img_width=320
img_size=170

def crop_image(arr,x1,y1,x2,y2):
    return arr[y1:y2,x1:x2];


def get_positive_images(l):
    for i in l:
        f=open('./dataset/'+i+'/'+i+'_loc.csv')
        bounding_boxes=f.read().split('\n')[1:-1]
        position = get_position(i);
        pos[position]=[]
        for j in bounding_boxes:
            im,x1,y1,x2,y2=j.split(',')
            img=io.imread('./dataset/'+im)
            img=color.rgb2grey(img)
            img=crop_image(img,int(x1),int(y1),int(x2),int(y2))
            img=transform.resize(img,(img_height,img_width))
            pos[position].append(img)
            # io.imsave('./sample/pos/'+im,img)
            print im

def IOU(predicted,actual):
	cord1 = [max(actual[0],predicted[0]),max(actual[1],predicted[1])]
	cord2 = [min(actual[0]+actual[2],predicted[0]+predicted[2]),min(actual[1]+actual[2],predicted[1]+predicted[2])]
	intersectionArea = (cord2[0]-cord1[0])*(cord2[1]-cord1[1])
	unionArea = actual[2]*actual[2] + predicted[2]*predicted[2] - intersectionArea
	return float(intersectionArea)/unionArea

def get_negative_images(l):
    for i in l:
        f=open('./dataset/'+i+'/'+i+'_loc.csv')
        bounding_boxes=f.read().split('\n')[1:-1]
        position = get_position(i);
        neg[position]=[]
        c=0
        for j in bounding_boxes:
            im,x1,y1,x2,y2=j.split(',')
            x1=int(x1);y1=int(y1);x2=int(x2);y2=int(y2)
            img=io.imread('./dataset/'+im)
            img=color.rgb2grey(img)
            size=x2-x1  #Assuming bounding box is a square
            for y in xrange(0,img_height-size,65):
                for x in xrange(0,img_width-size,65):
                    p=IOU([x,y,size],[x1,y1,size])
                    if(p>0.5):continue;
                    cr_img=crop_image(img,int(x),int(y),int(x+size),int(y+size))
                    cr_img=transform.resize(cr_img,(img_height,img_width))
                    neg[position].append(cr_img)
                    # io.imsave('./sample/neg/'+i+'/'+str(c)+".jpg",cr_img)
                    # io.imsave('./sample/neg/'+i+'/'+str(c)+"_"+str(p)+".jpg",cr_img)
                    c+=1
            print im,c

svm = svm.SVC(kernel='linear',C=10,probability=True);		        #Untrained SVM Classifier


def train(l):
    postive_features=[]
    negative_features=[]
    for i in l:
        position = get_position(i);
        for j in pos[position]:
			postive_features.append(feature.hog(j));
        for k in neg[position]:
            negative_features.append(feature.hog(k));

    X = postive_features + negative_features;
    Y = [1]*len(postive_features) + [0]*len(negative_features);
    svm.fit(X,Y);

def test(l):
    postive_features=[]
    negative_features=[]
    for i in l:
        position = get_position(i);
        for j in pos[position]:
			postive_features.append(feature.hog(j));
        for k in neg[position]:
            negative_features.append(feature.hog(k));

    X = postive_features + negative_features;
    Y = [1]*len(postive_features) + [0]*len(negative_features);
    return svm.score(X,Y)



l=l1+l2+l3+l4
l=["user_3","user_4"]
get_positive_images(l)
get_negative_images(l)

train(["user_3"])
print test(["user_4"])

# train(l2+l3+l4)
# s1=test(l1)
#
# train(l1+l3+l4)
# s2=test(l2)
#
# train(l1+l2+l4)
# s3=test(l3)
#
# train(l1+l2+l3)
# s4=test(l4)
#
# print s1,s2,s3,s4,float(s1+s2+s3+s4)/4.0
