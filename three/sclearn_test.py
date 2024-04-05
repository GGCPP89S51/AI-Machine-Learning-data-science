from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(x,y,classifier,test_idx=None,resolution=0.02):

    markers = ('s','x','o','^','V')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min , x1_max = x[:,0].min() - 1 ,x[:,0].max(0)+1
    x2_min , x2_max = x[:,1].min() - 1 ,x[:,1].max(0)+1
    xx1 , xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha = 0.3 , cmap = cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.xlim(xx2.min(),xx2.max())

    for idx , cl in enumerate(np.unique(y)):
        plt.scatter(x= X[y == cl , 0 ], y=x[y==cl,1],
                    alpha=0.8,c=colors[idx],
                    marker=markers[idx],label = cl ,
                    edgecolors='black')
    if test_idx:
        X_test,Y_test = x[test_idx,:],y[test_idx]

        plt.scatter(X_test[:,0],X_test[:,1],
                    c=colors[idx],edgecolors='black',alpha=1.0,
                    linewidths=1,marker='o',
                    s=100,label = 'test set')
        
#提取套件內的iris資料
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
print('Class labels:',np.unique(y))

#將資料分為測試集合訓練集
'''
此部分將資料拆分 訓練集:測試集 為 7:3
random_state 為 是否要亂數切割
stratify 為 選擇類別
'''
X_train , X_test , Y_train , Y_test =  train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
print ('Labels counts in y:',np.bincount(y))
print ('Labels counts in y_train:',np.bincount(Y_train))
print ('Labels counts in y_test:',np.bincount(Y_test))

#梯度下降法

#資料標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#訓練模型
'''
eta 為學習速率
random_state 為 是否要亂數訓練
'''
ppn = Perceptron(eta0=0.1,random_state=1)
ppn.fit(X_train_std,Y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (Y_test != y_pred).sum())

#計算正確率 和 失誤率
print('Accurany: %.3f' % accuracy_score(Y_test,y_pred))
print('Accuracy: %.3f' % ppn.score(X_test_std,Y_test))

x_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((Y_train,Y_test))
plot_decision_regions(x=x_combined_std,
                      y=y_combined,
                      classifier=ppn,
                      test_idx=range(105,150))
plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()