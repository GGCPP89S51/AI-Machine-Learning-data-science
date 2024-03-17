import numpy as np 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

'''
隨機梯度下降算法實現的自適應線性神經元（Adaline）類別。
這個類別實現了隨機梯度下降來更新權重，並在訓練過程中對輸入數據進行洗牌，以幫助提高模型的收斂性和泛化能力。
這種類型的神經元可以用於二分類問題，通過訓練過程來學習如何區分不同類別的樣本。
'''
#隨機自適應線性神經元物件
class AdalineSGD(object):
    #定義學習速率、迭代次數、洗牌、隨機種子
    def __init__(self, eta=0.01, n_iter=10,shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state

    '''
    初始化權重，然後執行指定數量的迭代。
    如果設置了洗牌標誌，它會對輸入數據進行洗牌操作。
    在每次迭代中，它會計算成本，更新權重，並將成本添加到成本列表中。
    最後，它會返回訓練後的模型。
    '''
    #訓練模型
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_=[]
        for i in range(self.n_iter):
            if self.shuffle:
                X,y=self._shuffle(X,y)
            cost = []
            for Xi,target in zip(X,y):
                cost.append(self._update_weights(Xi,target))
            avg_cost = sum(cost)/ len(y)
            self.cost_.append(avg_cost)
        return self
    
    '''
    如果模型的權重尚未初始化，它會使用輸入數據的特徵數來初始化權重。
    然後，它會檢查目標的形狀，如果目標是一維的，則認為只有一個樣本進行了訓練，因此會對該樣本調用 _update_weights 方法。
    否則，它會遍歷所有輸入和目標並逐個調用 _update_weights 方法。最後，它返回訓練後的模型。
    '''

    #增量式訓練
    def partial_fit(self,X,y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0]>1:
            for Xi,target in zip(X,y):
                self._update_weights(Xi,target)
        else:
            self._update_weights(X,y)
        return self
    
    #隨機排列
    def _shuffle(self,X,y):
        r = self.rgen.permutation(len(y))
        return X[r],y[r]
    
    #初始化模型的權重
    def _initialize_weights(self,m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0,scale=0.01,size=1+m)
        self.w_initialized=True

    #更新模型的權重
    def _update_weights(self,Xi,target):
        output = self.activation(self.net_input(Xi))
        error = (target-output)
        self.w_[1:]+=self.eta*Xi.dot(error)
        self.w_[0] += self.eta* error
        cost = 0.5*error**2
        return cost
    
    #網路輸入
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    #啟用設定
    def activation(self, X):
        return X
    
    #預測    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

#繪製決策區域的散點圖
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'X', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    X1_min, X1_maX = X[:, 0].min() - 1, X[:, 0].max() + 1
    X2_min, X2_maX = X[:, 1].min() - 1, X[:, 1].max() + 1
    XX1, XX2 = np.meshgrid(np.arange(X1_min, X1_maX, resolution),
                           np.arange(X2_min, X2_maX, resolution))
    Z = classifier.predict(np.array([XX1.ravel(), XX2.ravel()]).T)
    Z = Z.reshape(XX1.shape)
    plt.xlim(XX1.min(), XX1.max())
    plt.ylim(XX2.min(), XX2.max())

    for idX, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=cmap(idX),
                    marker=markers[idX],
                    label=cl,
                    edgecolors='black')


s = os.path.join(r'C:\Users\s0901\OneDrive\文件\GitHub\AI-Machine-Learning-data-science\iris\iris.data')
print('URL:', s)
df = pd.read_csv(s, header=None)

print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

X_std=np.copy(X)
X_std[:,0]=(X[:,0]-X[:,0].mean()) / X[:,0].std()
X_std[:,1]=(X[:,1]-X[:,1].mean()) / X[:,1].std()

ada = AdalineSGD(n_iter=15,eta=0.01,random_state=1)
ada.fit(X_std,y)
plot_decision_regions(X_std,y,classifier=ada)
plt.title('adaline - Stochastic Graddient Descent')
plt.xlabel('sepal length[standardized]')
plt.ylabel('petal length[standardized]')
plt.legend(loc ='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.tight_layout()
plt.show()