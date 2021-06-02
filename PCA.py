'''
Perform Principal component analysis and perform clustering using first 
3 principal component scores (both heirarchial and k mean clustering(scree plot or elbow curve) and obtain 
optimum number of clusters and check whether we have obtained same number of clusters with the original data 
(class column we have ignored at the begining who shows it has 3 clusters)df
'''

# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc

# load wine.csv as pandas dataframe
data = pd.read_csv('wine.csv')
# print(data.head())

# print(data.shape) # 178 rows, 14 columns

# print(data.columns)

# print(data.Type.value_counts())

# print(data.info())

# print(data.describe())

# standardisation
scaler = preprocessing.StandardScaler()
std_data = scaler.fit_transform(data.iloc[:, 1:])
print("Standardised data: \n", std_data)

# Create PCA object
pca=PCA()
pca.fit(std_data)
pca_data = pca.transform(std_data)
# pca_data = pca.fit_transform(std_data)
print("PCA Data: \n", pca_data)

# Checking percentage of explained variance ratio of all the principal components
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
print("Explained Variance Ratio:", per_var)

# Computing the cumulative of explained variance ratio
cum_variance = np.cumsum(np.round(pca.explained_variance_ratio_*100, decimals=1))
print("Cumulative Explained Variance Ratio:", cum_variance)

# Scree Plot
labels = ['PC'+str(x) for x in range(1, len(per_var)+1)] 
plt.figure(figsize=(10, 8))
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# Writing pca_data to a dataframe
pca_df = pd.DataFrame(pca_data, columns=labels)
print(pca_df)

# Visualizing first 3 principal components
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
ax.scatter3D(pca_df.PC1, pca_df.PC2, pca_df.PC3, c='red', alpha = 0.8)

plt.title("3D scatter plot")
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

# Scree plot on standardised data to find Optimal K
kmeans = KMeans()
plt.figure(figsize=(10, 8))
SSE = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(std_data)
    SSE.append(kmeans.inertia_) #criterion based on which K-means clustering works
plt.plot(range(1, 11), SSE)
plt.title('The Elbow Method for Standardised data')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Scree plot on first 3 Principal components to find Optimal K
kmeans = KMeans()
plt.figure(figsize=(10, 8))
SSE = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(pca_df.loc[:, ['PC1', 'PC2', 'PC3']])
    SSE.append(kmeans.inertia_) #criterion based on which K-means clustering works
plt.plot(range(1, 11), SSE)
plt.title('The Elbow Method - K-means with PCA Clustering')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Both the elbow curves look similar having elbow like bend at k=3 

# Creating KMeans object with 3 clusters
kmeans_pca3 = KMeans(n_clusters = 3)
kmeans_pca3.fit(pca_df.loc[:, ['PC1', 'PC2', 'PC3']]) #using first 3 principal components as per the problem statement
# print(kmeans_pca3.labels_)

pca_clusters = kmeans_pca3.predict(pca_df.loc[:, ['PC1', 'PC2', 'PC3']])
# Adding labels to the original dataset
data.insert(1, 'pca_clusters', pca_clusters+1)
print(data.pca_clusters.value_counts())

# Creating KMeans object with 3 clusters
kmeans_std_data = KMeans(n_clusters = 3)
kmeans_std_data.fit(std_data) # using standardised original data
std_data_clusters = kmeans_std_data.predict(std_data)

data.insert(2, 'std_data_clusters', std_data_clusters+1)
print(data.std_data_clusters.value_counts())

# We got the same clustering results

# Hierarchical clustering
plt.figure(figsize=(10, 6))
plt.title("Dendogram for Standardised Data - Ward")
dend = shc.dendrogram(shc.linkage(std_data, method='ward'))

plt.figure(figsize=(10, 6))
plt.title("Dendogram for first 3 principal components - Ward")
dend = shc.dendrogram(shc.linkage(pca_df.loc[:, ['PC1', 'PC2', 'PC3']], method='ward'))
plt.show()




