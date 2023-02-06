#=========================================================================================================================================================
#Importing of libraries
#=========================================================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection, neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#=========================================================================================================================================================
#Importing of Datasets
#=========================================================================================================================================================
#flood = pd.read_csv("India_Floods_Inventory.csv") #Not too sure how to predict with this dataset
data = pd.read_csv("kerala.csv")

#Exploratory Data Analysis
data.head()
data.apply(lambda x:sum(x.isnull()), axis = 0) #Checking for null values in any column
data['FLOODS'].replace(['YES','NO'],[1,0],inplace = True) #Making the Yes/No result numerical --> 1 = Yes, 0 = No
data.head() #Checking that the change was correct

#=========================================================================================================================================================
#Preparing the dataset for use
#=========================================================================================================================================================
#Extracting the desired variable columns for prediction
flood_variables = data.iloc[:, 1:14] #Extracting Columns 1 to 13 (Last index not included using iloc)
flood_variables.head()

#Extracting the results column for prediction
flood_results = data.iloc[:, -1] #Extracting the last column
flood_results.head()

#Scaling variables to be between 0 and 1
minmax = preprocessing.MinMaxScaler(feature_range = (0,1))
minmax.fit(flood_variables).transform(flood_variables)

#Creating train-test set
random.seed(3407)
variables_train, variables_test, results_train, results_test = train_test_split(flood_variables, flood_results, test_size = 0.2)

#Checking train and test set
variables_train.head()
variables_test.head()
results_train.head()
results_test.head()

#=========================================================================================================================================================
#Prediction using Different Models
#=========================================================================================================================================================
#KNN Classifier
clf = neighbors.KNeighborsClassifier()
knn_clf = clf.fit(variables_train, results_train)
knn_results_predict = knn_clf.predict(variables_test)
print("Predicted chances of flood:\n", knn_results_predict) #Predicted results
print("Actual chances of flood:\n", results_test.values) #Actual results

#Evaluating Model Accuracy
knn_accuracy = cross_val_score(knn_clf, variables_test, results_test, cv = 3, scoring = 'accuracy', n_jobs = -1)
knn_accuracy.mean() #Accuracy = 70.8%
print("Accuracy score: %f"%(accuracy_score(results_test, knn_results_predict) * 100)) #Score = 95.8%
print("Recall score: %f"%(recall_score(results_test, knn_results_predict) * 100)) #Score = 93.3%
print("Roc score: %f"%(roc_auc_score(results_test, knn_results_predict) * 100)) #Score = 96.7%

#=========================================================================================================================================================
#Logistic Regression
variables_train_std = minmax.fit_transform(variables_train)
variables_test_std = minmax.transform(variables_test)
lr = LogisticRegression()
lr_clf = lr.fit(variables_train_std, results_train)
lr_results_predict = lr_clf.predict(variables_test_std)
print("Predicted chances of flood:\n", lr_results_predict) #Predicted results
print("Actual chances of flood:\n", results_test.values) #Actual results

#Evaluating Model Accuracy
lr_accuracy = cross_val_score(lr_clf, variables_test_std, results_test, cv = 3, scoring = 'accuracy', n_jobs = -1)
lr_accuracy.mean() #Accuracy = 62.5%
print("Accuracy score: %f"%(accuracy_score(results_test, lr_results_predict) * 100)) #Score = 95.8%
print("Recall score: %f"%(recall_score(results_test, lr_results_predict) * 100)) #Score = 93.3%
print("Roc score: %f"%(roc_auc_score(results_test, lr_results_predict) * 100)) #Score = 96.7%

#=========================================================================================================================================================
#Decision Tree Classification
dtc_clf = DecisionTreeClassifier()
dtc_clf.fit(variables_train, results_train)
dtc_results_predict = dtc_clf.predict(variables_test)
print("Predicted chances of flood:\n", dtc_results_predict) #Predicted results
print("Actual chances of flood:\n", results_test.values) #Actual results

#Evaluating Model Accuracy
dtc_clf_accuracy = cross_val_score(dtc_clf, variables_test_std, results_test, cv = 3, scoring = "accuracy", n_jobs = -1)
dtc_clf_accuracy.mean() #Accuracy = 70.8%
print("Accuracy score: %f"%(accuracy_score(results_test, dtc_results_predict) * 100)) #Score = 70.8%
print("Recall score: %f"%(recall_score(results_test, dtc_results_predict) * 100)) #Score = 73.3%
print("Roc score: %f"%(roc_auc_score(results_test, dtc_results_predict) * 100)) #Score = 70.0%

#=========================================================================================================================================================
#Random Forest Classification
rfc = RandomForestClassifier(max_depth = 3, random_state = 0)
rfc_clf = rfc.fit(variables_train, results_train)
rfc_results_predict = dtc_clf.predict(variables_test)
print("Predicted chances of flood:\n", rfc_results_predict) #Predicted results
print("Actual chances of flood:\n", results_test.values) #Actual results

#Evaluating Model Accuracy
rfc_clf_accuracy = cross_val_score(rfc_clf, variables_test_std, results_test, cv = 3, scoring = "accuracy", n_jobs = -1)
rfc_clf_accuracy.mean() #Accuracy = 62.5%
print("Accuracy score: %f"%(accuracy_score(results_test, rfc_results_predict) * 100)) #Score = 70.8%
print("Recall score: %f"%(recall_score(results_test, rfc_results_predict) * 100)) #Score = 73.3%
print("Roc score: %f"%(roc_auc_score(results_test, rfc_results_predict) * 100)) #Score = 70.0%
#=========================================================================================================================================================
#Comparing all the models
models = []
names = []
scores = []
models.append(('KNN', neighbors.KNeighborsClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))

for name, model in models:
    model.fit(variables_train, results_train)
    results_predicted = model.predict(variables_test)
    scores.append(accuracy_score(results_test, results_predicted))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

#Visualising the model accuracies
axis = sns.barplot(x = 'Name', y = 'Score', data = tr_split)
axis.set(xlabel = 'Classifier', ylabel = 'Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show() #KNN and LR same accuracy

#=========================================================================================================================================================
#Reference: https://www.kaggle.com/code/mukulthakur177/flood-prediction-model/notebook