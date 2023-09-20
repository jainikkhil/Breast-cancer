#import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

#load the data  
df= pd.read_csv('data.csv')
#df.head(7)
#df.shape

#drop columns of empty values
df.dropna(axis =1)

#count the number if malignent or benin
df['diagnosis'].value_counts()

#look at data which column need to be encoded 
#df.dtypes

#encode the catagoricla data values
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
labelencoder_y.fit_transform(df.iloc[:,1].values)

#create a pairplot
#sns.pairplot(df.iloc[:,1:6],hue ='diagnosis')

#get the correlation of the columns
#df.iloc[:,1:12].corr()

#visualise the correlation
#plt.figure(figsize=(10,10)) 
#sns.heatmap(df.iloc[:,1:12].corr(), annot = True , fmt ='.0%')

#split the data into independent(x) and (y) data sets
x=df.iloc[:,2:31].values
y=df.iloc[:,1].values

#split dataset into 75%training and 25% testing 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.25, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#creat function for model
def models(x_train,y_train):

#logistic regression
  from sklearn.linear_model import LogisticRegression
  log= LogisticRegression(random_state=0)
  log.fit(x_train,y_train)

#decisiontree
  from sklearn.tree import DecisionTreeClassifier
  tree=DecisionTreeClassifier(criterion = 'entropy',random_state=0)
  tree.fit(x_train,y_train)

#Random Forest CLASSIFOER
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
  forest.fit(x_train,y_train)


#print the models accuracy on the training data
  print('[0] logistic accuracy',log.score(x_train,y_train))
  print('[1] decision accuracy',tree.score(x_train,y_train))
  print('[2] randomforest accuracy',forest.score(x_train,y_train))

  return log,tree,forest


#getting all models
model = models(x_train,y_train)

#test  model on test data on confusion matrix
from sklearn.metrics import confusion_matrix
for i in range(len(model)):
 cm = confusion_matrix(y_test,model[i].predict(x_test))
 tp= cm[0][0]
 tn=cm[1][1]
 fp=cm[1][0]
 fn=cm[0][1]

 print(cm)
 print('testing accuracy',(tp +tn)/(tp +tn +fp + fn))

 #another way to get metrics of the model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(classification_report(y_test,model[0].predict(x_test)))
print(accuracy_score(y_test,model[0].predict(x_test)))




#check if giving correct results or not
pred = model[2].predict(x_test)
print(pred)

print()
print(y_test)

