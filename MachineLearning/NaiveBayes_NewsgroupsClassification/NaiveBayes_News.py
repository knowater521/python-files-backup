

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.naive_bayes import MultinomialNB
 

news = fetch_20newsgroups(subset='all')
#查看数据规模和细节
#
print(len(news.data))
print(news.data[0])

x_train,x_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)

vec = CountVectorizer()
x_train = vec.fit_transform(x_train) #将文本转化为向量
x_test = vec.transform(x_test)
#print(x_train)
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
y_predict = mnb.predict(x_test)

print('The accuracy of Naive Bayes classifier is ',mnb.score(x_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))

