

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC  #导入线性SVM
digits = load_digits()
print(digits.data.shape)

x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
#初始化线性SVM
lsvc = LinearSVC()

#进行模型训练
#
lsvc.fit(x_train,y_train)

#预测
#
y_predict = lsvc.predict(x_test)

print('The accuracy of linear SVC is ',lsvc.score(x_test,y_test))

print(classification_report(y_test,y_predict,target_names=digits.target_names.astype(str)))
