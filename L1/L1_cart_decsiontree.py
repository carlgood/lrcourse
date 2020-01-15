from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# 加载数据
digits = load_digits()
data = digits.data

#数据探索
print(data.__len__())
print(data.shape)
# 查看第一幅图像
print(digits.images[1])
# 第一幅图像代表的数字含义
print(digits.target[1])
#将第一幅图像显示出来
plt.gray()
plt.title('Handwritten Digits')
plt.imshow(digits.images[1])
plt.show()

#分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=3)

#采用Z-score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

#创建CART分类器
dt = DecisionTreeClassifier()
dt.fit(train_ss_x, train_y)
predict_y = dt.predict(test_ss_x)
print('CART准确率：%0.5lf' % accuracy_score(predict_y, test_y))

