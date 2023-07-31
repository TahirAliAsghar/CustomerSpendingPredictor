#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import sys


# In[44]:


data_path = 'C:\\Users\\HP\\Desktop\\Mall_Customers AI.csv'
df = pd.read_csv(data_path)


# In[45]:


df.head()


# In[46]:


df.pop('location')
df.head()


# In[5]:


df.shape #200 rows 5 columns


# In[ ]:





# In[47]:


df.drop('CustomerID', axis=1, inplace = True)
df.head()


# In[49]:


df.isnull().sum()#no null data


# In[59]:


cor = df.corr()
sns.set(font_scale=1.4)
plt.figure(figsize=(9,8))
sns.heatmap(cor, annot=True, cmap='plasma')
plt.tight_layout()
plt.show() #older customers have less income and therefore spend less money.


# In[88]:


# The fit_transform() method is used to fit the data into a model and transform it into a form that is more suitable for the model in a single step


from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
df['Gender'] = le.fit_transform(df.iloc[:,0])
df ['location'] = le.fit_transform(df.iloc[:,0])
df.head()




# In[89]:


df['location'] = df['location'].fillna(1)
df.head(10)


# In[90]:


df.pop('location')
df.head()


# In[69]:


spending_score_male = 0
spending_score_female = 0

for i in range(len(df)):
    if df['Gender'][i] == 1:
        spending_score_male = spending_score_male + df['Spending Score (1-100)'][i]
    if df['Gender'][i] == 0:
        spending_score_female = spending_score_female + df['Spending Score (1-100)'][i]


print(f'Males Spending Score  : {spending_score_male}')
print(f'Females Spending Score: {spending_score_female}')
#Since the spending scores are very close to each other, the difference between the total spending scores is the difference between the number of male and female customers, but this difference is not serious.
#Considering all this, it would be meaningless to choose a gender-based target audience. ✓


# In[111]:


x = df[['Age','Spending Score (1-100)']].values
plt.scatter(df['Age'],df['Spending Score (1-100)'])
plt.xlabel('Age')
plt.ylabel("Spending Score (1-100)")


# In[79]:


# finding optimum number of clusters
from sklearn.cluster import KMeans
wcss_list = []

for i in range(1,11):
    kmeans_test = KMeans(n_clusters = i, init ='k-means++', random_state=40)
    kmeans_test.fit(x)
    wcss_list.append(kmeans_test.inertia_)

plt.figure(figsize=(9,6))
plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method', color='red',fontsize='23')
plt.xlabel('Number of clusters')
plt.xticks(np.arange(1,11))
plt.ylabel('WCSS')
plt.show()


# In[80]:


kmeans = KMeans(n_clusters = 4, init ='k-means++', random_state=88)
y_kmeans = kmeans.fit_predict(x)


# In[ ]:





# In[ ]:





# In[ ]:





# In[110]:


plt.scatter(df.Age,df['Spending Score (1-100)'])
plt.xlabel('Age')
plt.ylabel("Spending Score (1-100)")

km=KMeans(n_clusters=4)
y_predicted = km.fit_predict(df[['Age','Spending Score (1-100)']])
y_predicted

df['cluster']=y_predicted
df.head(10)


# In[112]:


km.cluster_centers_

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
df4=df[df.cluster==3]
plt.scatter(df1.Age,df1['Spending Score (1-100)'],color='green')
plt.scatter(df2.Age,df2['Spending Score (1-100)'],color='red')
plt.scatter(df3.Age,df3['Spending Score (1-100)'],color='black')
plt.scatter(df4.Age,df4['Spending Score (1-100)'],color='blue')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 85, c = 'brown', label = 'Centroids')

plt.title("Customers' Clusters")
plt.xlabel('Age', color='red')
plt.ylabel('Spending Score', color='red')
plt.legend()
plt.show()



# In[113]:


X = df.drop('Spending Score (1-100)',axis=1).values
y = df['Spending Score (1-100)'].values


# In[211]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=5)


# In[212]:


from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,11)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    
    knn = KNeighborsClassifier(n_neighbors=k)
    
    
    knn.fit(X_train, y_train)
    
    
    train_accuracy[i] = knn.score(X_train, y_train)
    
   
    test_accuracy[i] = knn.score(X_test, y_test) 


# In[213]:


plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[214]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[215]:


knn.fit(X_train,y_train)


# In[216]:


knn.score(X_test,y_test)


# In[217]:


from sklearn.metrics import confusion_matrix


# In[218]:


y_pred = knn.predict(X_test)


# In[219]:


confusion_matrix(y_test,y_pred)


# In[220]:


predicted_value= knn.predict([[19,21]])
print('spending score:', predicted_value)


# In[221]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




