# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:31:18 2021

@author: Sathiya vigraman M
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split

dir(datasets) # to find namei of datasets

iris = datasets.load_iris() #import data from sklearn
#iris = datasets.load_iris(as_frame = True)

iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)

iris_y = pd.DataFrame(iris.target)

iris_y.replace(dict(enumerate(iris.target_names)), inplace=True)

iris_df.isnull().sum() 

x_train, x_test, y_train, y_test = train_test_split(iris_df, iris_y, test_size = 0.30)  #split multiple DF in a same commend


-------------Classfication-------------------

x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


-----------KNN----------------


from sklearn.neighbors import KNeighborsClassifier as KNC

#best k value --- sqrt (n/2)

x_knn = KNC()   #modal
x_knn.fit(X = x_train, y = y_train) #fitting

x_knn.score(X = x_train, y = y_train) #training score

x_knn.score(X = x_test, y = y_test) #testing score

-----Confusion Matrix-----
from sklearn.metrics import confusion_matrix

accuracy = []

for i in range(1, 106): #120 is exculded
    iris_knn = KNC(n_neighbors = i)
    iris_knn.fit(X = x_train, y = y_train)
    accuracy.append(iris_knn.score(X = x_test, y = y_test))


maximum_indices = np.where(accuracy==max(accuracy))

accuracy_df = pd.DataFrame(accuracy)
accuracy_df['kvalue'] = range(1, 106)
accuracy_df.columns = ['acuracy', 'kvalue']
accuracy_df[accuracy_df['acuracy'] == accuracy_df.acuracy.max()]#finding both colunms
accuracy_df.kvalue[accuracy_df['acuracy'] == accuracy_df.acuracy.max()]#finding particular colunm



plt.plot(range(1, 106), accuracy, color='green', linestyle='-', marker='*', markerfacecolor='red', markersize=5)
plt.title('accuracy for K value')
plt.xlabel('K value')
plt.ylabel('accuracy')

plt.bar(range(1, 106), accuracy, color='green', )
plt.title('accuracy for K value')
plt.xlabel('K value')
plt.ylabel('accuracy')




#knn for 
#plot   x axis k value ---y ---accurcy
#Adaboost modal

-------------Decision tree Class------------

from sklearn.tree import DecisionTreeClassifier as DTC

iris_decic_tree = DTC()

iris_decic_tree.fit(X = x_train, y = y_train)

iris_decic_tree.score(X = x_train, y = y_train)

iris_decic_tree.score(X = x_test,  y = y_test)



----------Random forest classfi------------

from sklearn.ensemble import RandomForestClassifier as RFC

iris_random_forest = RFC()

iris_random_forest.fit(X = x_train, y = y_train)

iris_random_forest.score(X = x_train, y = y_train)

iris_random_forest.score(X = x_test,  y = y_test)

#hyper parameter tuning = max_depth in Random forest, Decision tree, k value in KNN


#Bagging and Boosting

------------AdaBoostClassifier--------------

from sklearn.ensemble import AdaBoostClassifier as ABC

iris_ada_boost = ABC()

iris_ada_boost.fit(X = x_train, y = y_train)

iris_ada_boost.score(X = x_train, y = y_train)

iris_ada_boost.score(X = x_test,  y = y_test)












