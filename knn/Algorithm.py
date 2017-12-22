import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA,KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def Classify_Euclidean(inX,dataSet,labels,k):
  dataSetSize = len(dataSet)
  diffMat = np.tile(inX,(dataSetSize,1)) - dataSet       #np.tile(A,B) 重复A，B次
  sqDiffMat = diffMat ** 2
  sqDistances = sqDiffMat.sum(axis=1)     #按行相加
  distances = sqDistances ** 0.5
  sortedDistIndicies = distances.argsort()  #返回排序的顺序号的索引，从0开始，第一个数是最小的数的索引，第二个数是次小数的索引，以此类推
  classCount={} 
  for i in range(k):
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel[0]] = classCount.get(voteIlabel[0],0) + 1
  sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]

def Classify_nthDistance(inX,dataSet,labels,k,n):
  dataSetSize = len(dataSet)
  diffMat = np.tile(inX,(dataSetSize,1)) - dataSet       #np.tile(A,B) 重复A，B次
  sqDiffMat = abs(diffMat) ** n
  sqDistances = sqDiffMat.sum(axis=1)     #按行相加
  distances = sqDistances ** (1/n)
  sortedDistIndicies = distances.argsort()  #返回排序的顺序号的索引，从0开始，第一个数是最小的数的索引，第二个数是次小数的索引，以此类推
  classCount={} 
  for i in range(k):
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel[0]] = classCount.get(voteIlabel[0],0) + 1
  sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]
   
def GetAccuracy(training_data,training_labels,test_data,test_labels,MaxK):
  length = len(test_data)
  predicts = np.zeros((length,1))
  accuracy = np.zeros((MaxK+1,1)) 
  for j in range(1,MaxK+1):
    count = 0
    for i in range(length):
        predicts[i] = Classify_Euclidean(test_data[i],training_data,training_labels,j)
        if predicts[i] == test_labels[i]:
          count = count + 1
        accuracy[j] = count/length
  return accuracy
def Define_set(n,data,label):
  end = len(data)
  training_data0 = data[0:end:5]
  training_labels0 = label[0:end:5]
  training_data1 = data[1:end:5]
  training_labels1 = label[1:end:5]
  training_data2 = data[2:end:5]
  training_labels2 = label[2:end:5]
  training_data3 = data[3:end:5]
  training_labels3 = label[3:end:5]
  training_data4 = data[4:end:5]
  training_labels4 = label[4:end:5]
  if n==0:
    x_test = training_data0
    y_test = training_labels0
    x_train = np.vstack((training_data1,training_data2,training_data3,training_data4))
    y_train = np.vstack((training_labels1,training_labels2,training_labels3,training_labels4))
  elif n == 1:
    x_test = training_data1
    y_test = training_labels1
    x_train = np.vstack((training_data0,training_data2,training_data3,training_data4))
    y_train = np.vstack((training_labels0,training_labels2,training_labels3,training_labels4))
  elif n == 2:
    x_test = training_data2
    y_test = training_labels2
    x_train = np.vstack((training_data0,training_data1,training_data3,training_data4))
    y_train = np.vstack((training_labels0,training_labels1,training_labels3,training_labels4))
  elif n ==3 :
    x_test = training_data3
    y_test = training_labels3
    x_train = np.vstack((training_data0,training_data1,training_data2,training_data4))
    y_train = np.vstack((training_labels0,training_labels1,training_labels2,training_labels4))
  else:
    x_test = training_data4
    y_test = training_labels4
    x_train = np.vstack((training_data0,training_data1,training_data2,training_data3))
    y_train = np.vstack((training_labels0,training_labels1,training_labels2,training_labels3))
    
  return x_test,y_test,x_train,y_train
    
    
#不用交叉验证，K取不同值时的准确率
def Without_folds(MaxK,data,label):
 accuracy_without_folds = GetAccuracy(data,label,data,label,MaxK)
 return accuracy_without_folds
 # plt.plot(accuracy_without_folds[1:MaxK + 2])
 # plt.title("Accuracy Without 5-fold ")
 # plt.xlabel("k-value")
 # plt.ylabel("accuracy")
 # np.savetxt('Accuracy Without 5-fold.csv',accuracy_without_folds)
 print("'Accuracy Without 5-fold.csv' is saving...")
#使用5折交叉验证的平均准确率
def With_folds(MaxK,data,label):
  temp = np.zeros((MaxK+1,1))
  for i in range(5):
    x_test,y_test,x_train,y_train = Define_set(i,data,label)
    accuracy = GetAccuracy(x_train,y_train,x_test,y_test,MaxK)
    temp = temp+accuracy
  temp = temp/5
  return temp
   
def WithPCA(MaxK,data,label):
  X = data
  y = label
  pca = PCA(n_components = 6)
  X_r = pca.fit(X).transform(X)
  accuracy = With_folds(MaxK,X_r,y)
  print("first 6 components:%s"%str(pca.explained_variance_ratio_))
  return accuracy

def BoxPlot(data):
  df = pd.DataFrame()
  for i in range(13):
    str1 = str(i+1)
    df[str1] = data[:,i]
  plt.figure()
  plt.boxplot(x=df.values,labels=df.columns)
  plt.title('boxplot')
  plt.xlabel('features')
  plt.ylabel('value')
  
def BoxPlotWithNormal(data):
  aver_array = np.mean(data,axis=0)   #对列求均值
  data = data/aver_array
  df = pd.DataFrame()
  for i in range(13):
    str1 = str(i+1)
    df[str1] = data[:,i]
  plt.figure()
  plt.boxplot(x=df.values,labels=df.columns)
  plt.title('boxplot after normalization')
  plt.xlabel('features')
  plt.ylabel('value')
  return data

def GetAccuracy_nthDistance(training_data,training_labels,test_data,test_labels,MaxK,n):
  length = len(test_data)
  predicts = np.zeros((length,1))
  accuracy = np.zeros((MaxK+1,1)) 
  for j in range(1,MaxK+1):
    count = 0
    for i in range(length):
        predicts[i] = Classify_nthDistance(test_data[i],training_data,training_labels,j,n)
        if predicts[i] == test_labels[i]:
          count = count + 1
        accuracy[j] = count/length
  return accuracy

def ChangeDistance(n,MaxK,data,label):
  temp = np.zeros((MaxK+1,1))
  for i in range(5):
    x_test,y_test,x_train,y_train = Define_set(i,data,label)
    accuracy = GetAccuracy_nthDistance(x_train,y_train,x_test,y_test,MaxK,n)
    temp = temp+accuracy
  temp = temp/5
  return temp


def WithKernelPCA(MaxK,data,label):
  X = data
  y = label
  kpca = KernelPCA(n_components = 6,kernel='poly', fit_inverse_transform=True, gamma=10)
  X_kpca = kpca.fit_transform(X)
  X_back = kpca.inverse_transform(X_kpca)
  accuracy = With_folds(MaxK,X_back,y)
  return accuracy

def WithLDA(MaxK,data,label):
  X = data
  y = label
  lda = LinearDiscriminantAnalysis(n_components = 6)
  lda.fit(X,y)
  X_lda = lda.transform(X)
  accuracy = With_folds(MaxK,X_lda,y)
  return accuracy
#def With_folds(MaxK,data,label):
#  acc_array = np.zeros((5,MaxK + 1))
#  acc_ave = []
#  for i in range(5):
#    x_test,y_test,x_train,y_train = Define_set(i,data,label)
#    accuracy = GetAccuracy(x_train,y_train,x_test,y_test,MaxK)
#    acc_array[i] = accuracy.ravel()
#    acc_ave.append(mean(accuracy))
#  accuracy = acc_array[acc_ave.index(max(acc_ave))]
#  return accuracy