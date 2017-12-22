

from sklearn import linear_model #导入线性模型

model = linear_model.LinearRegression()   #选择线性回归模型



from sklearn import datasets   #导入数据集

boston = datasets.load_boston()  #选择boston数据集
# print(boston)
y = boston.target



from sklearn.model_selection import cross_val_predict

predicted = cross_val_predict(model, boston.data, y, cv=10)  #交叉验证，sklearn中的流程
                                                                    #model 是我们选择要进行交叉验证的模型
                                                                    #data 是数据
                                                                    #target是数据的目标值
															 #cv是可选项，是数据折叠的总次数（k折）	
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))  #创建采用黑色边框的散点
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4) #ax.plot计算两个轴的最大最小值，k--表示线形，
															 #lw=4 代表宽度，然后给x,y加上标签
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig1,ax = plt.subplots()        #必须这样用
ax.plot([0,50],[0,50],'k--',lw=4)


