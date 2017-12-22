# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:05:18 2017

@author: pfWu
"""

import numpy as np
import matplotlib.pyplot as plt
import Algorithm



filename=r"proj1_data\wine.data"
dataset = np.loadtxt(filename)
labels = dataset[:,0].reshape((178,1))
datas = dataset[:,1:14]
MaxK = 70

def bootstrap(data):
  num = 500
  for i in range(num):
    index = int(np.random.rand(1)*178)
    data = np.vstack((data,data[index]))
  y = data[:,0].reshape((num+178,1))
  x = data[:,1:14]
  aver_array = np.mean(x,axis=0)
  x = x/aver_array
  accuracy = Algorithm.Without_folds(MaxK,x,y)
  accuracy_ = Algorithm.Without_folds(MaxK,datas,labels)
  plt.figure()
  plt.plot(accuracy[1:MaxK + 2],color='r',label='bootstap')
  plt.plot(accuracy_[1:MaxK + 2],color = 'b',label='original')
  plt.title("Accuracy of bootstrap and original ")
  plt.xlabel("k-value")
  plt.ylabel("accuracy")
  plt.legend(loc='upper right')


def Impfolds():
  accuracy = Algorithm.With_folds(MaxK,datas,labels)  
  plt.figure()
  plt.plot(accuracy[1:MaxK + 2])
  Str = "Accuracy With 5-fold"
  plt.title(Str)  
  plt.xlabel("k-value")
  plt.ylabel("accuracy")
  return accuracy
#  np.savetxt('Accuracy With 5-fold.csv',accuracy)
#  print("'Accuracy With 5-fold.csv' is saving...")

def ImpPCA(data):
  accuracy_pca = Algorithm.WithPCA(MaxK,data,labels)           ##PCA
  plt.figure()
  plt.plot(accuracy_pca[1:MaxK + 2])
  Str = "Accuracy With PCA and 5-fold"
  plt.title(Str)  
  plt.xlabel("k-value")
  plt.ylabel("accuracy")
#  np.savetxt('Accuracy With PCA.csv',accuracy_pca)
#  print("'Accuracy With PCA.csv' is saving...")
  return accuracy_pca
def ImpKernelPCA():
  accuracy_kpca = Algorithm.WithKernelPCA(MaxK,datas,labels)
  plt.figure()
  plt.plot(accuracy_kpca[1:MaxK + 2],color='r',label='kernelPCA')
  plt.plot(accuracy_pca[1:MaxK+2],color='b',label='PCA')
  Str = "Accuracy of kPCA and KernelPCA"
  plt.title(Str)  
  plt.xlabel("k-value")
  plt.ylabel("accuracy")
  plt.legend(loc ='upper right')
  return accuracy_kpca

def ImpLDA(data):
  accuracy_lda = Algorithm.WithLDA(MaxK,data,labels)
  plt.figure()
  plt.plot(accuracy_lda[1:MaxK + 2],color='r',label='lda')
  plt.plot(accuracy_pca[1:MaxK+2],color='b',label='PCA')
  plt.plot(accuracy[1:MaxK+2],color='g',label='original')
  Str = "Accuracy of original,PCA and LDA"
  plt.title(Str)  
  plt.xlabel("k-value")
  plt.ylabel("accuracy")
  plt.legend(loc ='upper right')
  return accuracy_lda

def DifferentN():
  plt.figure()
  length_N = 10
  accuracy_array=np.zeros((length_N,1))
  x_axis = np.zeros((length_N,1))
  for n in range(1,length_N):
    accuracy_nthDistance = Algorithm.ChangeDistance(n+1,MaxK,data_normalized,labels)
    accuracy_array[n] = np.mean(accuracy_nthDistance)
    x_axis[n] = n+1
    plt.plot(accuracy_nthDistance[1:MaxK + 2],label='n=%d'%(n+1))
  plt.legend(loc='lower left')
  Str = "Accuracy with differernt n"
  plt.title(Str)  
  plt.xlabel("k-value")
  plt.ylabel("accuracy")
  plt.figure()
  plt.plot(x_axis[1:length_N],accuracy_array[1:length_N])
  plt.title('Mean accuracy of differernt n')
  plt.xlabel("n")
  plt.ylabel("accuracy")

def Normalized():
  accuracy_normalized = Algorithm.With_folds(MaxK,data_normalized,labels)
  plt.figure()
  plt.plot(accuracy_normalized[1:MaxK + 2])
  Str = "Accuracy After normalized"
  plt.title(Str)  
  plt.xlabel("k-value")
  plt.ylabel("accuracy")

  plt.figure()
  plt.plot(accuracy[1:MaxK + 2],color='r',label='original data')
  plt.plot(accuracy_normalized[1:MaxK + 2],color='b',label='normalized data')
  plt.legend(loc = 'center right')
  plt.title('Accuracy Comparison')
  plt.xlabel("k-value")
  plt.ylabel("accuracy")
  
def Boxs():
  Algorithm.BoxPlot(datas)
  data_normalized = Algorithm.BoxPlotWithNormal(datas)
  return data_normalized
if __name__ == "__main__":
  accuracy = Impfolds()
  accuracy_pca = ImpPCA(datas);
  change = accuracy - accuracy_pca    #检查PCA后准确率的变化
  data_normalized = Boxs();
  Normalized()
  DifferentN()
#  accuracy_kpca = ImpKernelPCA()
  accuracy_pca = ImpPCA(data_normalized)
  accuracy_lda = ImpLDA(data_normalized)
  bootstrap(dataset)

  

  
  
  
  