import numpy as np 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#自適應線性神經元物件
class AdalineGD(object):
    #定義學習速率、迭代次數、隨機種子
    def __init__(self, eta=0.01, n_iter=50, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    '''
    通過 np.random.RandomState(self.random_state) 初始化了一個隨機數生成器 rgen，以確保每次運行的結果是可重現的。
    使用 rgen.normal 方法初始化了模型的權重 self.w_，將其初始化為均值為 0，標準差為 0.01 的正態分佈。
    初始化了一個空列表 self.cost_，用於保存每一次迭代的成本。
    進行了 self.n_iter 次迭代的訓練過程：
    對於每次迭代，計算了模型的淨輸入 net_input 和激活函數的輸出 output。
    計算了每個樣本的誤差 errors。
    根據誤差和輸入特徵，使用梯度下降法更新了權重。
    計算了每次迭代的平均成本，並將其添加到 self.cost_ 列表中。
    最後返回了 self，以實現方法鏈式調用。
    '''
    #訓練模型
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    #網路輸入
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    #啟用設定
    def activation(self, X):
        return X
    
    #預測    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
'''
它首先計算了決策區域的邊界，通過使用 np.meshgrid 函數在特徵空間中創建一個網格，然後使用分類器對每個網格點進行預測。
對於每個類別，它使用不同的顏色和標記來繪製散點圖，並通過 alpha 參數設置了點的透明度。
最後，它設置了 x 和 y 軸的範圍，並添加了圖例。
'''    
#繪製決策區域的散點圖
def plot_decision_regions(x, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0],
                    y=x[y == cl, 1],
                    alpha=0.8,
                    c=cmap(idx),
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')


s = os.path.join(r'C:\Users\s0901\OneDrive\文件\GitHub\AI-Machine-Learning-data-science\iris\iris.data')
print('URL:', s)
df = pd.read_csv(s, header=None)

print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0, 2]].values
'''
ada1 和 ada2 是分別使用學習速率為 0.01 和 0.0001 初始化的 AdalineGD 實例。
使用每個 Adaline 實例對訓練數據進行擬合，並將其成本函數值記錄在 cost_ 屬性中。
在每個子圖上，使用 plot 函數繪製了成本函數值的對數以及執行次數的關係。這樣的對數尺度通常用於更好地可視化成本函數的變化。
設置 x 軸和 y 軸的標籤以及子圖的標題。
最後通過 plt.show() 顯示圖像。
'''
fig , ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(x, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-errors)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(x, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-errors)')
ax[1].set_title('Adaline - Learning rate 0.0001')        
plt.show()

'''
對特徵進行標準化。對於每個特徵，計算其平均值和標準差，然後將每個值減去平均值並除以標準差，從而將特徵的範圍標準化到均值為0，方差為1的標準正態分佈。
創建一個 AdalineGD 的實例 ada，設置迭代次數為15，學習速率為0.01。
使用標準化後的特徵 x_std 和目標變量 y 對 Adaline 模型進行擬合。
使用 plot_decision_regions 函數繪製決策區域的散點圖，並添加標題、x軸和y軸的標籤以及圖例。
顯示決策區域的散點圖。
使用 plt.plot 函數繪製每個迭代次數的平均成本函數值，以觀察訓練過程中成本函數的變化。
顯示平均成本函數值的圖形。
'''
x_std=np.copy(x)
x_std[:,0]=(x[:,0]-x[:,0].mean()) / x[:,0].std()
x_std[:,1]=(x[:,1]-x[:,1].mean()) / x[:,1].std()

ada = AdalineGD(n_iter=15,eta=0.01)
ada.fit(x_std,y)
plot_decision_regions(x_std,y,classifier=ada)
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