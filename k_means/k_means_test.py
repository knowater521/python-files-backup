import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics

file_name =r"F:\desktop\数据(1)\数据\DEAP\subject_video.txt"
y,x = np.loadtxt(file_name,unpack='true')

x=x.reshape((1216,1))
y=y.reshape((1216,1))

#X,y = make_blobs(n_samples = 1000 ,n_features = 2,
#	centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2],random_state = 9)
#plt.scatter(X[:,0],X[:,1],marker='o')
#plt.show()


y_pred = KMeans(n_clusters=32, random_state=9).fit_predict(x)
#plt.scatter(X[:, 0], X[:, 1], c=y_pred)
#plt.show()
#print(metrics.calinski_harabaz_score(X, y_pred)  )