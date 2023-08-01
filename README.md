import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pip install plotly

pip install cufflinks

from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import cufflinks as cf



df= pd.read_csv(r"C:\Users\User\Desktop\healthcare-dataset-stroke-data.csv")

df

df.tail()

df.shape

df.size

#### Overview of data

df.info()

df.isnull().sum()

#### Handling missing values

df=df.fillna(0)

df.info()

df.isnull().sum()


#### Describing the data

df.describe()

#### Analysis and visualization

#plot graph for individual columns

df

sns.countplot(df['gender'])

sns.countplot(df['hypertension'])

sns.countplot(df['heart_disease'])

sns.countplot(df['ever_married'])

sns.countplot(df['work_type'])

sns.countplot(df['Residence_type'])

sns.countplot(df['avg_glucose_level'])

sns.countplot(df['smoking_status'])

sns.countplot(df['stroke'])

df['stroke'].value_counts()

df['stroke'].value_counts().plot(kind='pie',autopct='%0.2f%%')

df['gender'].value_counts().plot(kind='pie',autopct='%0.2f%%')
df['gender'].value_counts()

#### Bar Graph

df['gender'].value_counts().plot(kind='bar')
plt.xlabel('gender')
plt.ylabel('heart_disease')
plt.title('Male & female who are having heart problems')
plt.show()

#### Distplot

sns.distplot(df["age"],hist=True)

#### Histogram

plt.figure(figsize=(8,8))
plt.hist(df['avg_glucose_level'])
plt.title('avg_glucose_level')
plt.xlabel('avg_glucose_level')
plt.show()

#### countplot

plt.figure(figsize=(7,7))
sns.countplot(df['age'],hue=df['heart_disease'])
plt.show()

df['heart_disease'].value_counts()

plt.figure(figsize=(7,7))
sns.countplot(df['gender'],hue=df['smoking_status'])
plt.show()

df['smoking_status'].value_counts()

plt.figure(figsize=(7,7))
sns.countplot(df['heart_disease'],hue=df['Residence_type'])
plt.show()

df['Residence_type'].value_counts()

#### Box Plot

plt.figure(figsize=(7,7))
sns.boxplot(data=df,x='stroke',y='age',palette='viridis')
plt.show()


#### pairplot

plt.figure(figsize=(8,8))
sns.pairplot(df)
plt.show()

#### Heatmap

df.corr()

plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,linewidth=1.0)

df.info()

#### Handling Categorical Variables

from sklearn import preprocessing
le= preprocessing.LabelEncoder()

df['gender']=le.fit_transform(df['gender'])

df['ever_married']=le.fit_transform(df['ever_married'])

df['work_type']=le.fit_transform(df['work_type'])

df['Residence_type']=le.fit_transform(df['Residence_type'])

df['smoking_status']=le.fit_transform(df['smoking_status'])

df.info()

df

### Features & Target

features = df.iloc[:,:-1]
target = df.iloc[:,-1]

features

target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()

from sklearn.svm import SVC
svm=SVC()

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()

from sklearn.metrics import accuracy_score,classification_report

cr=classification_report
ac=accuracy_score

def mymodel(model):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(classification_report(y_test,y_pred))
    return model

mymodel(logreg)

svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print(classification_report(y_test,y_pred))
ac(y_test,y_pred)

mymodel(knn)

mymodel(dt)

dt1=DecisionTreeClassifier(max_depth=10)
mymodel(dt1)


