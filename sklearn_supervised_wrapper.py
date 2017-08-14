# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:43:33 2017

@author: Jay Bandlamudi
"""

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class sklearn_supervised_wrapper:
        def __init__(self,data_path,target_name,classifier):
            self.data=data_path
            self.target=target_name
            self.classifier=classifier
        def load_data(self,split_ratio):
            seed = 2017
            self.dataset = pandas.read_csv(self.data)
            array = self.dataset.values
            X = array[:,0:4]
            Y = array[:,4]
            validation_size = split_ratio #0.20
            self.X_train, self.X_validation, self.Y_train, self.Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

            
        def describe_data(self):
            print "First five rows of the dataset "
            print(self.dataset.head(5))
            
            print "Class distribution in the dataset "
            print (self.dataset.groupby(self.target).size())
            
            print "Univariate plots"
            self.dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
            plt.show()
            
            print "Multivariate plots"
            scatter_matrix(self.dataset)
            plt.show()

        def train_model(self):
            model_dict={
                    'LR':('LR', LogisticRegression()),
                    'LDA':('LDA', LinearDiscriminantAnalysis()),
                    'KNN':('KNN', KNeighborsClassifier()),
                    'CART':('CART', DecisionTreeClassifier()), 
                    'NB':('NB', GaussianNB()),
                    'SVM':('SVM', SVC())
                          }
            models = []
            if self.classifier=='ALL':
                for k in model_dict.keys():
                    models.append(model_dict[k])
            else :
                for cls in self.classifier:
                    models.append(model_dict[self.classifier])
            results = []
            names = []
            for name, model in models:
            	kfold = model_selection.KFold(n_splits=10, random_state=2017)
            	cv_results = model_selection.cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring=scoring)
            	results.append(cv_results)
            	names.append(name)
            	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            	print(msg)
            # Compare Algorithms
            fig = plt.figure()
            fig.suptitle('Algorithm Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(results)
            ax.set_xticklabels(names)
            plt.show()

            
            
'''            
# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

dataset.delete('class')

print(dataset.head(20))

	
# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)




models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
    
    
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
'''