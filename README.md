# ds

# Ex1
```
import pandas as pd
df=pd.read_csv('/content/SAMPLEDS.csv')
df.head()
df.tail()
df.info()
df.describe()
df.shape
df.isnull().sum()
New_df=df.dropna()
New_df
mn=df.TOTAL.mean()
mn
df.head()
for x in df.index:
    if df.loc[x,"AVG"]>100:
        df.drop(x,inplace=True)
df
```
# EX2
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/content/titanic_dataset.csv')
print(df.head())
df.info()
df.set_index("PassengerId",inplace=True)
df.describe()
df.nunique()
df.shape
df["Survived"].value_counts()
person=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
person
sns.countplot(data=df,x="Survived")
df.Pclass.unique()
df.rename(columns ={'Sex': 'Gender'}, inplace = True)
df
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=0.7,color="Lightgreen")
sns. catplot(x='Survived', hue="Gender",data=df, kind = "count" )
df.boxplot(column="Age", by="Survived")
sns.scatterplot(x=df['Age'],y=df['Fare'])
sns.jointplot(x="Age",y="Fare",data=df)
fig, ax1= plt.subplots(figsize=(8,5))
pt=sns.boxplot (ax=ax1,x='Pclass',y='Age' ,hue= 'Gender', data=df)
sns. catplot(data=df, col = "Survived",x = "Gender", hue="Pclass", kind = "count" )
corr=df.corr()
sns.heatmap(corr,annot=True)
sns.pairplot(df)
```
# EX3 To read the given data and perform Feature Encoding and Transformation process and save the data to a file.
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

df['bo2']=e1.fit_transform(df[["ord_2"]])
df

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2

pd.get_dummies(df2,columns=["nom_0"])

pip install --upgrade category_encoders

from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
fb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb

from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

df.skew()

np.log(df["Highly Positive Skew"])

np.reciprocal(df["Moderate Positive Skew"])

np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Positive Skew"])

df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

# EX4 To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file.
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
data.isnull().sum()
missing=data[data.isnull().any(axis=1)]
missing
data2=data.dropna(axis=0)
data2
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
data2
new_data=pd.get_dummies(data2, drop_first=True)
new_data
columns_list=list(new_data.columns)
print(columns_list)
features=list(set(columns_list)-set(['SalStat']))
print(features)
y=new_data['SalStat'].values
print(y)
x=new_data[features].values
print(x)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
data.shape
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
tips.time.unique()
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
# EX5
```
import matplotlib.pyplot as plt
x_val = [0,1,2,3,4,5]
y_val = [0,1,4,9,16,25]
plt.plot(x_val,y_val)
plt.show()

import matplotlib.pyplot as plt
x = [1,2,3]
y = [2,4,1]
plt.plot(x,y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('My first graph')
plt.show()

import matplotlib.pyplot as plt
x1 = [1,2,3]
y1 = [2,5,3]
plt.plot(x1,y1,label = 'line 1')
x2 = [1,2,3]
y2 = [3,1,6]
plt.plot(x2,y2,label = 'line 2')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title("Two lines on the same graph")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x = [1,2,3,4,5]
y1 = [10,12,14,16,18]
y2 = [5,7,9,11,13]
y3 = [2,4,6,8,10]
plt.fill_between(x,y1,color = 'blue')
plt.fill_between(x,y2,color = 'orange')
plt.stackplot(x,y1,y2,y3,labels = ['line1','line2','line3'])
plt.legend(loc = 'upper left')
plt.title('Stacked line charts')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
val = [2,4,7,3]
names = ['A','B','C','D']
plt.bar(names, val,color = 'purple')
plt.show()
import matplotlib.pyplot as plt
import numpy as np
ages = [2,6,4,12,13,12,16,18,18,19,26,24,39,34,45,42,54,56,90,56,86,79]
range = (0,100)
bins = 10
plt.hist(ages,bins,range,color='green',histtype='bar',rwidth=0.8)
plt.xlabel('age')
plt.ylabel('no of people')
plt.title('histogram')
plt.show()
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
data=np.random.normal(loc=0,scale=1,size=100)
data
fig,ax=plt.subplots()
ax.boxplot(data)
ax.set_xlabel("data")
ax.set_ylabel("values")
ax.set_title("box plot")
import matplotlib.pyplot as plt
activities=['eat','sleep','work','play']
slices=[3,7,8,6]
colors=['r','y','g','b']
plt.pie(slices,labels = slices,colors=colors,startangle=90,shadow = True,explode = (0,0,0.1,0),radius=1.2,autopct='%1.1f%%')
plt.legend()
plt.show()
```
# EX6
```
import pandas as pd
df=pd.read_csv("/content/titanic_dataset.csv")
df.head()
import seaborn as sns
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[3,6,2,7,1]
sns.lineplot(x=x,y=y)
plt.title('Line Plot')
x=[1,2,3,4,5]
y1=[3,5,2,6,1]
y2=[1,6,4,3,8]
y3=[5,2,7,1,4]
sns.lineplot(x=x,y=y1)
sns.lineplot(x=x,y=y2)
sns.lineplot(x=x,y=y3)
plt.title('Multi Line Plot')
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked',y='Fare',data=df,palette='rainbow')
plt.title("Fare Of Passenger By Embarked Town")
sns.scatterplot(x="Age", y="Fare", data=df)
plt.title('Scatterplot of Age vs Fare')
sns.scatterplot(x="Age", y="Fare", size="Pclass", data=df, sizes=(30, 200))
plt.title('Bubble Chart of Age vs Fare, Size by Passenger Class')
sns.histplot(data=df,x="Pclass",hue="Survived",kde=True)
sns.boxplot(x='Pclass',y='Age',data=df,palette='rainbow')
plt.title("Age By Passenger Class")
sns.violinplot(x="Pclass", y="Fare", data=df)
plt.title('Violin Plot of Fare by Passenger Class')
sns.kdeplot(data=df['Age'], shade=True)
plt.title('Density Plot of Passenger Ages')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Titanic Dataset')
plt.show()
```
