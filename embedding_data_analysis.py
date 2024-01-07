import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import umap

#========================================================================================================================================================   
# 对embedding后的数据进行UMAP降维与聚类分析

# 加载 npz 数据
data_path = '/content/drive/My Drive/data.npz'
data = np.load(data_path)
X = data['X']
y = data['Y']

# UMAP 降维
umap_reducer = umap.UMAP()
X_reduced = umap_reducer.fit_transform(X)
y_reduced = umap_reducer.fit_transform(y)

# KMeans 聚类
kmeans_X = KMeans(n_clusters=6)  
kmeans_y = KMeans(n_clusters=100)

X_clusters = kmeans_X.fit_predict(X_reduced)
y_clusters = kmeans_y.fit_predict(y_reduced)

# 可视化降维和聚类结果
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=X_clusters, cmap='Spectral', s=5)
plt.title('UMAP projection of Cell-molecular coding', fontsize=24)

plt.subplot(1, 2, 2)
plt.scatter(y_reduced[:, 0], y_reduced[:, 1], c=y_clusters, cmap='Spectral', s=5)
plt.title('UMAP projection of gene differencial expression', fontsize=24)

plt.show()

#====================================================================================================================================================
# 对embedding后的数据进行相关性分析


# 定义聚类的数量
k = 100  

# 对 y 进行 K-means 聚类
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(y.T)  # 注意：对 y 的转置进行聚类

# 计算每个聚类的均值
cluster_means = np.array([y[:, clusters == i].mean(axis=1) for i in range(k)]).T

# 确保 X 和 cluster_means 在样本数量上是一致的
assert X.shape[0] == cluster_means.shape[0], "The number of samples must be the same in X and cluster_means."

# 计算 X 和 cluster_means 的相关性矩阵
X_corr_with_clusters = np.corrcoef(X, cluster_means, rowvar=False)  # rowvar=False 表明行代表样本

# 从相关性矩阵中提取 X 与 cluster_means 之间的相关性部分
# 这部分位于矩阵的右上角
num_features = X.shape[1]
num_clusters = k  # 之前定义的聚类数 k
corr_matrix_X_clusters = X_corr_with_clusters[:num_features, num_features:num_features+num_clusters]

# 将相关性矩阵的相关部分转换为 DataFrame
corr_df = pd.DataFrame(corr_matrix_X_clusters,
                       index=[f'Feature_{i}' for i in range(num_features)],
                       columns=[f'Cluster_{j}_Mean' for j in range(num_clusters)])

# 绘制热图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=False, cmap='coolwarm')
plt.title('Correlation analysis of cell-small molecule coding and gene expression data')
plt.show()