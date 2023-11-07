#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tslearn


# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import cdist_dtw


# In[27]:


df_normal = pd.read_csv('C:\\Users\\Gjirafa\\Alberti\\AlbZa\\DS Lab2\\Dataset\\SWaT_Dataset_Normal_v0.csv')
df_attack = pd.read_csv('C:\\Users\\Gjirafa\\Alberti\\AlbZa\\DS Lab2\\Dataset\\SWaT_Dataset_Attack_v0 - Copy.csv')


# In[28]:


# Convert timestamps to datetime 
df_normal['Timestamp'] = pd.to_datetime(df_normal['Timestamp'])


# In[39]:


df_normal.head()


# In[38]:


df_normal.dtypes


# In[52]:


df_attack = pd.read_csv('C:\\Users\\Gjirafa\\Alberti\\AlbZa\\DS Lab2\\Dataset\\SWaT_Dataset_Attack_v0 - Copy.csv')


# In[53]:


df_attack['Timestamp'] = pd.to_datetime(df_attack['Timestamp'], format='mixed')


# In[54]:


df_attack['Timestamp'] = df_attack['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')


# In[37]:


df_attack.head()


# In[40]:


# Drop unnecessary columns
columns_to_drop = ['AIT201', 'P201', 'FIT601', 'P601', 'P602', 'P603', 'Normal/Attack']
df_normal.drop(columns_to_drop, axis=1, inplace=True)
df_attack.drop(columns_to_drop, axis=1, inplace=True)


# In[41]:


# Print the shape of the DataFrame (number of rows and columns)
print(f'Shape of normal dataset: {df_normal.shape}')
print(f'Shape of attack dataset: {df_attack.shape}')

print()

# Print the column names of the DataFrame
print(f'Columns of normal dataset: {df_normal.columns}')
print(f'Columns of attack dataset: {df_attack.columns}')


# In[42]:


df_normal = df_normal.set_index('Timestamp')
df_attack = df_attack.set_index('Timestamp')


# In[43]:


# Standardize the data separately for normal and attack datasets
scaler_normal = StandardScaler()
scaler_attack = StandardScaler()


# In[44]:


df_normal = scaler_normal.fit_transform(df_normal)
df_attack = scaler_attack.fit_transform(df_attack)


# In[45]:


df_normal = pd.DataFrame(df_normal)
df_attack = pd.DataFrame(df_attack)

# Split the dataset into training and testing
train_data = df_normal  # Use the first 7 days for training
test_data = pd.concat([df_normal, df_attack])  # Combine both datasets for testing


# In[46]:


# Create Time Series K-Means model with DTW metric
n_clusters = 2
model = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', verbose=True)

# Train the model on normal operation data
model.fit(train_data)


# In[49]:


pip install joblib


# In[50]:


import joblib

model_filename = 'C:\\Users\\Gjirafa\\Alberti\\AlbZa\\DS Lab2\\Dataset\\anomaly_detectiontimeseries_kmeans_model.pkl'
joblib.dump(model, model_filename)


# In[51]:


loaded_model = joblib.load(model_filename)


# In[58]:


labels = loaded_model.predict(df_attack)


# In[65]:


import matplotlib.pyplot as plt

# Visual Inspection
# Plot some examples of time series data and highlight detected anomalies
anomalies = df_attack[labels == 1]  # Assuming label 1 represents anomalies
normal = df_attack[labels == 0]  # Assuming label 0 represents normal data


# In[66]:


# Plot anomalies in red and normal data in blue
plt.figure(figsize=(10, 6))
for series in anomalies:
    plt.plot(series, color='red', alpha=0.7)
for series in normal:
    plt.plot(series, color='blue', alpha=0.3)
plt.title("Anomaly Detection Results")
plt.show()


# In[ ]:


# Statistical Measures (Silhouette Score)
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(df_attack, labels)
print(f"Silhouette Score: {silhouette_avg}")


# In[ ]:




