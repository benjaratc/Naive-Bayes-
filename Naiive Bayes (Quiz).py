#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# 1.โหลด csv เข้าไปใน Python Pandas

# In[2]:


df = pd.read_csv('../Desktop/DataCamp/Glass.csv')
df


# 2. เขียนโค้ดแสดง หัว10แถว ท้าย10แถว และสุ่ม10แถว

# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.sample(10)


# 3. เช็คว่ามีข้อมูลที่หายไปไหม สามารถจัดการได้ตามความเหมาะสม

# In[6]:


df.isnull().sum()


# 4. ใช้ info และ describe อธิบายข้อมูลเบื้องต้น

# In[7]:


df.info()


# In[8]:


df.describe()


# 5. ใช้ pairplot ดูความสัมพันธ์เบื้องต้นของ features ที่สนใจ

# In[9]:


sns.pairplot(data = df)


# 6. ใช้ displot เพื่อดูการกระจายของแต่ละคอลัมน์

# In[10]:


sns.distplot(df['RI'], kde = False )


# In[11]:


sns.distplot(df['Na'], kde = False )


# In[12]:


sns.distplot(df['Si'], kde = False )


# In[13]:


sns.distplot(df['Ba'], kde = False )


# In[14]:


sns.distplot(df['Fe'], kde = False )


# 7. ใช้ heatmap ดูความสัมพันธ์ของคอลัมน์ที่สนใจ

# In[15]:


plt.figure(figsize = (12,8))
sns.heatmap(df.corr(), annot = df.corr())


# 8. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation สูงสุด

# In[16]:


plt.figure(figsize = (12,8))
sns.scatterplot(data = df, x = 'RI', y = 'Ca')


# 9. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation ต่ำสุด

# In[17]:


plt.figure(figsize = (12,8))
sns.scatterplot(data = df, x = 'Mg', y = 'Type')


# 10. สร้าง histogram ของ feature ที่สนใจ

# In[18]:


plt.figure(figsize = (12,8))
plt.hist(df['Mg'])


# 11. สร้าง box plot ของ features ที่สนใจ

# In[19]:


plt.figure(figsize = (12,8))
sns.boxplot(data = df, x = 'Type', y = 'Mg')


# In[20]:


plt.figure(figsize = (12,8))
sns.boxplot(data = df, x = 'Type', y = 'Ba')


# In[21]:


plt.figure(figsize = (12,8))
sns.boxplot(data = df, x = 'Type', y = 'Al')


# 13. ทำ Data Visualization อื่นๆ (แล้วแต่เลือก)

# In[22]:


sns.countplot(df['Type'])


# 14. พิจารณาว่าควรทำ Normalization หรือ Standardization หรือไม่ควรทั้งสองอย่าง พร้อมให้เหตุผล 

# ควรทำ Normalization เพราะ x ไม่เป็น normal distribution

# # Default

# In[23]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score,accuracy_score


# In[24]:


X = df.drop('Type', axis = 1)
y = df['Type']


# In[25]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 100)


# In[26]:


nb = GaussianNB()
nb.fit(X_train,y_train)


# In[27]:


predicted1 = nb.predict(X_test)
predicted1


# In[28]:


confusion_matrix(y_test,predicted1)


# In[29]:


print('accuracy score', accuracy_score(y_test,predicted1))    
print('precision score', precision_score(y_test,predicted1,average = 'micro'))
print('recall score', recall_score(y_test,predicted1,average = 'micro'))
print('f1 score', f1_score(y_test,predicted1,average = 'micro'))


# # Standardization

# In[30]:


X = df.drop('Type', axis = 1)
y = df['Type']


# In[31]:


sc_X = StandardScaler()
X1 = sc_X.fit_transform(X)


# In[32]:


X_train,X_test,y_train,y_test = train_test_split(X1,y, test_size = 0.2, random_state = 100)


# In[33]:


nb2 = GaussianNB()
nb2.fit(X_train,y_train)


# In[34]:


predicted2 = nb.predict(X_test)
predicted2


# In[35]:


confusion_matrix(y_test,predicted2)


# In[36]:


print('accuracy score', accuracy_score(y_test,predicted2))    
print('precision score', precision_score(y_test,predicted2,average = 'micro'))
print('recall score', recall_score(y_test,predicted2,average = 'micro'))
print('f1 score', f1_score(y_test,predicted2,average = 'micro'))


# # Normalization

# In[37]:


X = df.drop('Type', axis = 1)
y = df['Type']


# In[38]:


min_max_scaler = MinMaxScaler()


# In[39]:


X_minmax = min_max_scaler.fit_transform(X)
X_minmax


# In[40]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax,y, test_size = 0.2, random_state = 100)


# In[41]:


nb3 = GaussianNB()
nb3.fit(X_train,y_train)


# In[42]:


predicted3 = nb.predict(X_test)
predicted3


# In[43]:


confusion_matrix(y_test,predicted3)


# In[44]:


print('accuracy score', accuracy_score(y_test,predicted3))    
print('precision score', precision_score(y_test,predicted3,average = 'micro'))
print('recall score', recall_score(y_test,predicted3,average = 'micro'))
print('f1 score', f1_score(y_test,predicted3,average = 'micro'))


# 15. เลือกช้อยที่ดีที่สุดจากข้อ 14 (หรือจะทำทุกอันแล้วนำมาเปรียบเทียบก็ได้)

# Standardization ดีกว่า normalization

# 16. เลือกเฉพาะ features ที่สนใจมาเทรนโมเดล และวัดผลเปรียบเทียบกับแบบ all-features

# In[45]:


df.corr()


# In[46]:


X = df[['Mg','Al','Ba']]
y = df['Type']


# In[47]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 100)


# In[48]:


nb4 = GaussianNB()
nb4.fit(X_train,y_train)


# In[50]:


predicted4 = nb4.predict(X_test)
predicted4


# In[51]:


confusion_matrix(y_test,predicted4)


# In[52]:


print('accuracy score', accuracy_score(y_test,predicted4))    
print('precision score', precision_score(y_test,predicted4,average = 'micro'))
print('recall score', recall_score(y_test,predicted4,average = 'micro'))
print('f1 score', f1_score(y_test,predicted4,average = 'micro'))


# 17. ทำ Visualization ของค่า F1 Score ระหว่าง ผลลัพธ์ของ Normalization, Standardization ทั้งแบบก่อนและหลังการแบ่งข้อมูล

# In[53]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop(['Type'], axis =1),
                                                 df['Type'],
                                                 test_size = 0.2, 
                                                 random_state = 100
                                                )


# In[54]:


min_max_scaler2 = MinMaxScaler()


# In[55]:


X_train = min_max_scaler2.fit_transform(X_train)
X_train


# In[56]:


min_max_scaler3 = MinMaxScaler()


# In[57]:


X_test = min_max_scaler3.fit_transform(X_test)
X_test


# In[58]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[59]:


nb5 = GaussianNB()
nb5.fit(X_train,y_train)


# In[60]:


predicted5 = nb5.predict(X_test)
predicted5


# In[61]:


print('accuracy score', accuracy_score(y_test,predicted5))    
print('precision score', precision_score(y_test,predicted5,average = 'micro'))
print('recall score', recall_score(y_test,predicted5,average = 'micro'))
print('f1 score', f1_score(y_test,predicted5,average = 'micro'))


# In[62]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop(['Type'], axis =1),
                                                 df['Type'],
                                                 test_size = 0.2, 
                                                 random_state = 100
                                                )


# In[63]:


sc2 = StandardScaler()
sc3 = StandardScaler()


# In[64]:


X_train = sc2.fit_transform(X_train)
X_train


# In[65]:


X_test = sc3.fit_transform(X_test)
X_test


# In[66]:


nb6 = GaussianNB()
nb6.fit(X_train,y_train)


# In[67]:


predicted6 = nb6.predict(X_test)
predicted6


# In[68]:


confusion_matrix(y_test,predicted6)


# In[69]:


print('accuracy score', accuracy_score(y_test,predicted6))    
print('precision score', precision_score(y_test,predicted6,average = 'micro'))
print('recall score', recall_score(y_test,predicted6,average = 'micro'))
print('f1 score', f1_score(y_test,predicted6,average = 'micro'))


# In[70]:


data = {'Standardization after splitting': f1_score(y_test,predicted6,average = 'micro'),
        'Normalization after splitting' : f1_score(y_test,predicted5,average = 'micro'),
        'Standardization before splitting': f1_score(y_test,predicted2,average = 'micro'),
        'Normalization before splitting': f1_score(y_test,predicted3,average = 'micro')}
data


# In[71]:


Series1 = pd.Series(data = data)
Series1


# In[72]:


df1 = pd.DataFrame(Series1)
df1


# In[73]:


fig = plt.figure(figsize = (12,8))
sns.barplot(data = df1, x = df1.index, y = df1[0])
plt.ylabel('f1 score')


# 18. สามารถใช้เทคนิคใดก็ได้ตามที่สอนมา ใช้ Naïve Bayesแล้วให้ผลลัพธ์ที่ดีที่สุดที่เป็นไปได้ (ลองปรับ Parameter)

# ลองทำดูแล้วพอว่า ใช้ทุก independent variable Standardization ได้ผลดีที่สุด 
