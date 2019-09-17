# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn import svm
import glob
import os
import pickle
import numpy as np

#defining paths
curr_path = os.getcwd()
pos_feat_ph = os.path.join(curr_path, "pos_feat_ph")
neg_feat_ph = os.path.join(curr_path, "neg_feat_ph")
model_path=curr_path 

fds_hog = []
labels_hog = []
# Load the positive HOG features
for feat_path in glob.glob(os.path.join(pos_feat_ph,"*.feat")):
    fd = joblib.load(feat_path)
    fds_hog.append(fd)
    labels_hog.append(1)

# Load the negative HOG features
for feat_path in glob.glob(os.path.join(neg_feat_ph,"*.feat")):
    fd = joblib.load(feat_path)
    fds_hog.append(fd)
    labels_hog.append(-1)
    

clf=svm.SVC(kernel='linear',C=1.0)
print("Training a Linear SVM Classifier for HOG")
clf.fit(fds_hog, labels_hog)
file=open('model_hog.p','wp')    
pickle.dump(clf,file)
file.close()
print("Classifier saved to {}".format(model_path))
