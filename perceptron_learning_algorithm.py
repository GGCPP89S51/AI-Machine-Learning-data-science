import numpy as np 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
#感知器物件
class Percetron(object):
    #定義學習速率、迭代次數、隨機種子
    def __init__(self,eta=0.01,n_iter=50,random_state=None):
        self.eta =eta
        self.n_iter = n_iter
        self.random_state = random_state

    '''
    通過 np.random.RandomState(self.random_state) 初始化了隨機數生成器 rgen，以確保每次運行得到的隨機數序列一致。
    使用 rgen.normal 方法初始化了感知器的權重 self.w_，將其初始化為均值為0，標準差為0.01的正態分佈。
    初始化了一個空列表 self.errors，用於記錄每一次迭代中分類錯誤的次數。
    進行了 self.n_iter 次迭代訓練，每次迭代都遍歷訓練數據，計算並更新權重。
    在每次迭代中，對於訓練集中的每一個樣本，計算其預測值並根據預測結果對權重進行調整。
    將每次迭代中分類錯誤的次數添加到 self.errors 列表中，以便後續分析。
    最後返回 self，以實現方法鏈式調用。
    '''
    #訓練模型
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
            self.errors.append(errors)
        return self
    
    '''
    使用 np.dot 函數計算了輸入特徵向量 x 與權重向量 self.w_[1:] 的內積，這部分表示了特徵的線性組合。
    將偏置項 self.w_[0] 加到內積的結果中，得到最終的淨輸入值。
    '''
    #網路輸入
    def net_input(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]
    
    '''
    首先，使用 self.net_input(x) 計算輸入特徵 x 的淨輸入值。
    然後，使用 np.where 函數，將淨輸入值大於等於0的部分標記為1，否則標記為-1。
    最終返回預測結果，即類別標籤。
    '''
    #預測
    def predict(self,x):
        return np.where(self.net_input(x)>=0.0,1,-1)

#仔入鳶尾花資料
s = os.path.join(r'C:\Users\s0901\OneDrive\文件\GitHub\AI-Machine-Learning-data-science\iris\iris.data')
print('URL:', s)
df = pd.read_csv(s, header=None)

print(df.tail())

'''
df.iloc[0:100, 4].values 從 DataFrame 中選取了前 100 筆資料的第 4 欄（索引從 0 開始），並轉換為 NumPy 陣列。
np.where(y == 'Iris-setosa', -1, 1) 將 y 中所有等於 'Iris-setosa' 的標籤設為 -1，其餘設為 1，這是因為感知器算法通常要求標籤為 -1 和 1。
df.iloc[0:100, [0, 2]].values 從 DataFrame 中選取了前 100 筆資料的第 0 和第 2 欄，這些欄位通常用作特徵，並將其轉換為 NumPy 陣列。
'''
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0, 2]].values
'''
第一幅圖中，使用 plot.scatter 函數將資料集中的前 50 個樣本（setosa）和後 50 個樣本（versicolor）在二維特徵空間中繪製成散點圖。
其中，橫軸代表 sepal length，縱軸代表 petal length。紅色的圓圈代表 setosa，藍色的叉號代表 versicolor。
透過 plot.xlabel 和 plot.ylabel 設置了橫軸和縱軸的標籤，並透過 plot.legend 添加了圖例，最後使用 plot.show 顯示圖像。
'''
plot.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
plot.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')
plot.xlabel('sepal length[cm]')
plot.ylabel('petal length[cm]')
plot.legend(loc='upper left')
plot.show()
'''
第二幅圖中，使用 plot.plot 函數繪製了感知器在訓練過程中每個 epoch 的錯誤次數。
橫軸表示 epoch 數量，縱軸表示更新權重的次數。
透過 plot.xlabel 和 plot.ylabel 設置了橫軸和縱軸的標籤，最後使用 plot.show 顯示圖像。
'''
ppn = Percetron(eta=0.1,n_iter=10)
ppn.fit(x,y)
plot.plot(range(1,len(ppn.errors)+1),ppn.errors,marker='o')
plot.xlabel('Epochs')
plot.ylabel('Number of updates')
plot.show()

