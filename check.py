from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import skimage as ski
from skimage import io,transform,feature,color,data
import numpy as np


k=io.imread('./dataset/user_3/A0.jpg');
l, img=feature.hog(color.rgb2grey(k), visualise=True)
io.imsave('./a.jpg',img)
