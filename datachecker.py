import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 加载Parquet文件
df = pd.read_parquet('./data/open-problems-single-cell-perturbations/de_train.parquet')

# 按照细胞类型分割数据
df_nk_cells = df[df['cell_type'] == 'NK cells']
df_t_cells_cd4 = df[df['cell_type'] == 'CD4 cells']
df_t_cells_cd8 = df[df['cell_type'] == 'CD8 cells']
df_t_regulatory_cells = df[df['cell_type'] == 'Regulatory cells']
df_b_cells = df[df['cell_type'] == 'B cells']
df_myeloid_cells = df[df['cell_type'] == 'Myeloid cells']

# 对每个细胞类型分别做PCA降维和K-Means聚类

# NK cell
# 筛选基因表达数据
genes_nk_expression = df_nk_cells[:, 'A1BG':'ZZEF1']

# 标准化特征
scaler = StandardScaler()
genes_expression_scaled = scaler.fit_transform(genes_nk_expression)

# 应用PCA降维
pca = PCA(n_components=2)
principal_components = pca.fit_transform(genes_expression_scaled)

# 应用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)  
clusters = kmeans.fit_predict(principal_components)

# 可视化PCA和K-means结果
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('PCA and K-means Clustering of NK cells Gene Expression Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()


# CD4 cell
# 筛选基因表达数据
genes_cd4_expression = df_t_cells_cd4.loc[:, 'A1BG':'ZZEF1']

# 标准化特征
scaler = StandardScaler()
genes_expression_scaled = scaler.fit_transform(genes_nk_expression)

# 应用PCA降维
pca = PCA(n_components=2)
principal_components = pca.fit_transform(genes_expression_scaled)

# 应用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)  
clusters = kmeans.fit_predict(principal_components)

# 可视化PCA和K-means结果
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('PCA and K-means Clustering of CD4 cells Gene Expression Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()



# CD8 cell
# 筛选基因表达数据
genes_cd8_expression = df_t_cells_cd8.loc[:, 'A1BG':'ZZEF1']

# 标准化特征
scaler = StandardScaler()
genes_expression_scaled = scaler.fit_transform(genes_nk_expression)

# 应用PCA降维
pca = PCA(n_components=2)
principal_components = pca.fit_transform(genes_expression_scaled)

# 应用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)  
clusters = kmeans.fit_predict(principal_components)

# 可视化PCA和K-means结果
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('PCA and K-means Clustering of CD8 cells Gene Expression Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()


# Regulatory cells
# 筛选基因表达数据
genes_regulatory_expression = df_t_regulatory_cells.loc[:, 'A1BG':'ZZEF1']

# 标准化特征
scaler = StandardScaler()
genes_expression_scaled = scaler.fit_transform(genes_nk_expression)

# 应用PCA降维
pca = PCA(n_components=2)
principal_components = pca.fit_transform(genes_expression_scaled)

# 应用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)  
clusters = kmeans.fit_predict(principal_components)

# 可视化PCA和K-means结果
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('PCA and K-means Clustering of Regulatory cells Gene Expression Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# B cells
# 筛选基因表达数据
genes_b_expression = df_b_cells.loc[:, 'A1BG':'ZZEF1']

# 标准化特征
scaler = StandardScaler()
genes_expression_scaled = scaler.fit_transform(genes_nk_expression)

# 应用PCA降维
pca = PCA(n_components=2)
principal_components = pca.fit_transform(genes_expression_scaled)

# 应用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)  
clusters = kmeans.fit_predict(principal_components)

# 可视化PCA和K-means结果
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('PCA and K-means Clustering of B cells Gene Expression Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Myeloid cells
# 筛选基因表达数据
genes_myeloid_expression = df_myeloid_cells.loc[:, 'A1BG':'ZZEF1']

# 标准化特征
scaler = StandardScaler()
genes_expression_scaled = scaler.fit_transform(genes_nk_expression)

# 应用PCA降维
pca = PCA(n_components=2)
principal_components = pca.fit_transform(genes_expression_scaled)

# 应用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)  
clusters = kmeans.fit_predict(principal_components)

# 可视化PCA和K-means结果
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('PCA and K-means Clustering of Myeloid cells Gene Expression Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()
