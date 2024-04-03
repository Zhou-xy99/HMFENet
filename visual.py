# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 07:53:27 2021

@author: 1
"""
#首先为了省事，将训练神经网络得库全都导入进来

import torch
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import FactorAnalysis,FastICA,PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from collections import OrderedDict
import matplotlib.pyplot as plt

# 载入数据
df = pd.read_csv('gnn_dist_1_unseen.csv')
X = np.expand_dims(df.values[:, 0:134].astype(float), axis=2)
Y = df.values[:, 134]
#这里很重要，不把标签做成独热码得形式最终出的图空白一片
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = torch.nn.functional.one_hot(torch.tensor(Y_encoded))

X = X.reshape(119,134)#这个形状根据读取到得csv文件得的大小自行调试，8621是列数，此列数比CSV文件中少一行
# 加载数据
def get_data():
	"""
	:return: 数据集、标签、样本数量、特征数量
	"""
	#digits = datasets.load_digits(n_class=10)
	digits = 2
	data = X  # digits.data		# 图片特征
	label = Y  # digits.target		# 图片标签
	n_samples = 119 # 对应reshape中的行数
	n_features = 134  # 对应reshape中的列数
	return data, label, n_samples, n_features


# 对样本进行预处理并画图
def plot_embedding(data, label, title):
	"""
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
	font = {'family': 'Times New Roman'}

	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
	fig = plt.figure(figsize=(6, 5))		# 创建图形实例
	ax = plt.subplot(111)		# 创建子图，经过验证111正合适，尽量不要修改
	# 遍历所有样本
	colors = ['red', 'c', 'greenyellow', 'pink', 'yellowgreen', 'grey']
	# colors = ['violet', 'pink', 'skyblue', 'yellow', 'greenyellow', 'palevioletred', 'mediumorchid', 'c', 'orange', 'yellowgreen']
	# colors = ['violet', 'pink', 'skyblue', 'yellow', 'greenyellow']

	for i in range(data.shape[0]):
		# 在图中为每个数据点画出标签
		# plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 5),
		# 		 fontdict={'weight': 'bold', 'size': 9})
		if(str(label[i])=='0.0'):
			color = colors[0]
		elif(str(label[i]) == '1.0'):
			color = colors[1]
		elif (str(label[i]) == '2.0'):
			color = colors[2]
		elif (str(label[i]) == '3.0'):
			color = colors[3]
		elif (str(label[i]) == '4.0'):
			color = colors[4]
		elif (str(label[i]) == '5.0'):
			color = colors[5]
		# elif (str(label[i]) == '6.0'):
		# 	color = colors[6]
		# elif (str(label[i]) == '7.0'):
		# 	color = colors[7]
		# elif (str(label[i]) == '8.0'):
		# 	color = colors[8]
		# elif (str(label[i]) == '9.0'):
		# 	color = colors[9]
		plt.scatter(data[i, 0], data[i, 1], c=color, label=str(label[i]))

	plt.xticks()		# 指定坐标的刻度
	plt.yticks()
	# plt.title(title, fontsize=14)

	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), ['query(T72)','BMP2','BRDM2','ZIL131','D7','BTR70'], loc='lower right', prop = font)
	# plt.legend(by_label.values(), ['ZSU234','T62','BTR60','D7','ZIL131','ZSU234(proto)','BTR60(proto)','ZIL131(proto)','D7(proto)','T62(proto)'], loc='lower left', prop = font)
	# plt.legend(by_label.values(), ['D7','ZIL131','BTR60','T62','ZSU234'], loc='lower left', prop = font)
	plt.savefig('gnn_dist_unseen_new.pdf')
	# 返回值
	return fig


# 主函数，执行t-SNE降维
def main():
	data, label , n_samples, n_features = get_data()		# 调用函数，获取数据集信息
	print('Starting compute t-SNE Embedding...')
	ts = TSNE(perplexity=67, n_components=2, init='random', random_state=0) #11 17
	# ts = TSNE(n_components=2, random_state=0)
	# ts = PCA(n_components=2)
	# ts = FastICA(n_components=2)
	# ts = FactorAnalysis(n_components=2)
	# ts = LinearDiscriminantAnalysis(n_components=2)
	# t-SNE降维
	reslut = ts.fit_transform(data)
	# numpy.savetxt("2pot.csv", reslut, delimiter=',')
	# 调用函数，绘制图像
	fig = plot_embedding(reslut, label, 'Display of distribution')
	# 显示图像
	plt.show()


# 主函数
if __name__ == '__main__':
	main()

