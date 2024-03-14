import numpy as np 
class Percetron(object):
    def __init__(self,eta=0.01,n_iter=50,random_state=None):
        self.eta =eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self,x,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size=1+x.shape[1])
        self.errors = []
        for _ in range(self.n_iter):
            errors = 0
            for xi , target in zip(x,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors += int(update != 0.0)
        return self
    
    def net_input(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]
    
    def predict(self,x):
        return np.where(self.net_input(x)>=0.0,1,-1)
'''   
import matplotlib.pyplot as plt

# 定義虛擬資料集
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, -1, -1])

# 初始化感知器物件
perceptron = Percetron()

# 訓練感知器
perceptron.fit(X, y)

# 預測並輸出結果
print("預測結果:", perceptron.predict(X))

# 繪製決策邊界
plt.scatter(X[:2, 0], X[:2, 1], color='red', marker='o', label='Class -1')
plt.scatter(X[2:, 0], X[2:, 1], color='blue', marker='x', label='Class 1')
plt.xlabel('X1')
plt.ylabel('X2')

# 畫出決策邊界
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                       np.arange(x2_min, x2_max, 0.1))
Z = perceptron.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3)
plt.legend(loc='upper left')
plt.title('Perceptron')
plt.show()
'''
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 

