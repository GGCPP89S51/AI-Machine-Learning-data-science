import pandas as pd
from sklearn.datasets import load_iris

# 載入 iris 資料集
iris = load_iris()

# 創建 DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 將 DataFrame 匯出至 CSV 檔案
df.to_csv('iris_dataset.csv', index=False)

print("資料已匯出至 iris_dataset.csv")