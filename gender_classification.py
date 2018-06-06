# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:34:56 2018

@author: Jed Sophos
"""

from sklearn import tree, svm
from sklearn.linear_model import SGDClassifier


#import graphviz 

clf = tree.DecisionTreeClassifier()
clf1= SGDClassifier(loss="hinge", penalty="l2")
clf2= svm.SVC()


# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf1= clf1.fit(X, Y)
clf2= clf2.fit(X, Y)

prediction = clf.predict([[190, 80, 42]])
pred=clf1.predict([[190, 80, 42]])
predi=clf2.predict([[190, 80, 42]])
#proba=clf.predict_proba([[160, 70, 39]])


# CHALLENGE compare their reusults and print the best one!

print("decisiontreeclassifier", prediction)
print("sgd", pred)
print("svm", predi)
#print(proba)

#dot_data = tree.export_graphviz(clf, out_file='tree.dot') 
#graph = graphviz.Source(dot_data) 
#graph.render("gender classification") 