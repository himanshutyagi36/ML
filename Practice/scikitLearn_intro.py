from sklearn import datasets
from sklearn import svm
import pickle
iris = datasets.load_iris()
digits = datasets.load_digits()
# print(digits.data)
## to print the ground truth
# print digits.target

## access the original sample image
# print digits.images[0]

## gamma - kernel coefficient, C - Penalty parameter C of the error term 
clf = svm.SVC(gamma=.0001, C=100.0,kernel='rbf')

## [:-1] - creates a new array that contains all but the last element.
clf.fit(digits.data[:-1], digits.target[:-1])
clf.predict(digits.data[-1:])
# print clf

## Model persistence : we can save model using Python's buil-in pickle library.
# X,y = iris.data, iris.target
# s = pickle.dumps(clf)
# clf2 = pickle.load(X[0:1])
# print y[0]