import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer

# 数据加载
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
# 数据探索
# 查看train_data信息
pd.set_option('display.max_columns', None) #显示所有列
print('查看数据信息：列名、非空个数、类型等')
print(train_data.info())
print('-'*30)
print('查看数据摘要')
print(train_data.describe())
print('-'*30)
print('查看离散数据分布')
print(train_data.describe(include=['O']))
print('-'*30)
print('查看前5条数据')
print(train_data.head())
print('-'*30)
print('查看后5条数据')
print(train_data.tail())

# 使用平均年龄来填充年龄中的nan值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
# 使用票价的均值填充票价中的nan值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

print(train_data['Embarked'].value_counts())
# 使用登录最多的港口来填充登录港口的nan值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)
# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]
print('特征值')
print(train_features)

dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)
# 构造ID3决策树
clf = DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features, train_labels)

test_features=dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
# pred_labels = clf.predict(test_features)

# 得到决策树准确率(基于训练集)
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score准确率为 %.4lf' % acc_decision_tree)

# 使用K折交叉验证 统计决策树准确率
print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))

# 创建LR分类器
lr = LogisticRegression(solver='liblinear', multi_class='auto') #数据集比较小，使用liblinear，数据集大使用 sag或者saga
lr.fit(train_features, train_labels)

# 得到LR准确率(基于训练集)
print(30 * '--')
print(u'LR score准确率为 %.4lf' % round(lr.score(train_features,train_labels), 6))

# 使用K折交叉验证 统计LR准确率
print(u'LR cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(lr, train_features, train_labels, cv=10)))

# 创建LDA分类器
lda = LinearDiscriminantAnalysis()
lda.fit(train_features,train_labels)

# 得到LDA准确率(基于训练集)
print(30 * '--')
print(u'LDA score准确率为 %.4lf' % round(lda.score(train_features,train_labels), 6))

# 使用K折交叉验证 统计LR准确率
print(u'LDA cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(lda, train_features, train_labels, cv=10)))

# 创建贝叶斯分类器
gnb = GaussianNB()
gnb.fit(train_features,train_labels)

# 得到贝叶斯分类器准确率(基于训练集)
print(30 * '--')
print(u'贝叶斯 score准确率为 %.4lf' % round(gnb.score(train_features,train_labels), 6))

# 使用K折交叉验证 统计贝叶斯分类器准确率
print(u'贝叶斯 cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(gnb, train_features, train_labels, cv=10)))

# 创建SVM分类器
mysvm = svm.SVC(kernel='rbf', C=1.0, gamma='auto')
mysvm.fit(train_features, train_labels)

# 得到svm分类器准确率(基于训练集)
print(30 * '--')
print(u'svm score准确率为 %.4lf' % round(mysvm.score(train_features,train_labels), 6))

# 使用K折交叉验证 统计svm分类器准确率
print(u'svm cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(mysvm, train_features, train_labels, cv=10)))

# 创建KNN分类器
knn = KNeighborsClassifier()
knn.fit(train_features, train_labels)
# 得到knn分类器准确率(基于训练集)
print(30 * '--')
print(u'KNN score准确率为 %.4lf' % round(knn.score(train_features,train_labels), 6))

# 使用K折交叉验证 统计knn分类器准确率
print(u'KNN cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(knn, train_features, train_labels, cv=10)))


