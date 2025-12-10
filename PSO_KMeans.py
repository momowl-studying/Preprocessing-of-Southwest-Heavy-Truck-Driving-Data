import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyswarm import pso  # 使用pyswarm库进行PSO优化

# 读取Excel数据
# df = pd.read_excel('E:/recognition/实车数据/最佳特征选择集合pca.xlsx')  # 替换为您的Excel文件路径
df = pd.read_excel('E:/recognition/实车数据/片段划分.xlsx')
# X = df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']].values
X = df[['平均行驶速度', '速度标准差', '加速度标准差', '减速度标准差', '加速度平方的均值', '匀速时间占比']].values

# # PSO参数设置
# k = 3  # 聚类数
# num_particles = 30  # 粒子数量
# max_iter = 100  # 最大迭代次数
#
# # 定义适应度函数
# def kmeans_inertia(centers):
#     # 确保centers是2D数组，形状为(k,4)，因为PSO要求的是输入二维数据（样本数，特征数）
#     if centers.ndim == 1:
#         centers = centers.reshape(k, -1)  # 将1D数组转换为2D数组
#     kmeans = KMeans(n_clusters=k, init=centers, n_init=1, max_iter=100)
#     kmeans.fit(X)
#     return kmeans.inertia_
#
# # 定义PSO的边界
# lb = np.min(X, axis=0).reshape(1, -1)  # 形状为 (1, 4)
# ub = np.max(X, axis=0).reshape(1, -1)  # 形状为 (1, 4)
# lb = np.tile(lb, (k, 1))  # 形状为 (k, 4)
# ub = np.tile(ub, (k, 1))  # 形状为 (k, 4)
#
# # 运行PSO优化，返回best_centers，_表示其他返回值未使用
# best_centers, _ = pso(kmeans_inertia, lb.flatten(), ub.flatten(), swarmsize=num_particles, maxiter=max_iter)
#
# # 将1D数组转换为2D数组，形状为 (k, 4)
# best_centers = best_centers.reshape(k, X.shape[1])
#
# print("优化后的初始聚类中心（标准化）:")
# print(best_centers)

# 使用优化后的初始中心进行K-Means聚类，n_init=1只运行一次K-Means，而不是多次运行并选择最佳结果
kmeans = KMeans(n_clusters=3,  n_init=10, max_iter=300)
kmeans.fit(X)

# 获取聚类标签
labels = kmeans.labels_

# 将聚类标签添加回原始数据
df['Cluster'] = labels + 1

# 获取最终的聚类中心
# centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['平均行驶速度', '速度标准差', '加速度标准差', '减速度标准差', '加速度平方的均值', '匀速时间占比'])
centroids['Cluster'] = centroids.index + 1  # 如果聚类标签从1开始

# 为聚类中心添加标识列
centroids['Type'] = 'Centroid'

# 为原始数据添加标识列
df['Type'] = 'DataPoint'

# 在每个聚类中心之间添加空行
centroids_with_empty = pd.concat([centroids, pd.DataFrame([''] * centroids.shape[1], index=centroids.columns).T], ignore_index=True)

# 将原始数据和聚类中心合并
df_with_centroids = pd.concat([df, centroids_with_empty], ignore_index=True)
# 保存结果到新的Excel文件
df_with_centroids.to_excel('E:/recognition/实车数据/聚类工况标签.xlsx', index=False)  # 您可以根据需要更改文件名

print("聚类完成，结果已保存到 'E:/recognition/实车数据/聚类工况标签'")