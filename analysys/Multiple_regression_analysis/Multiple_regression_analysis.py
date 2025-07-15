import pandas as pd

import statsmodels.api as sm

# データの読み込み（例: CSVファイル）
# df = pd.read_csv('data.csv')

# サンプルデータ（適宜置き換えてください）
# 説明変数: X1, X2, X3 / 目的変数: Y
data = {
    'Y': [10, 12, 13, 15, 16],
    'X1': [1, 2, 3, 4, 5],
    'X2': [2, 1, 4, 3, 5],
    'X3': [5, 3, 2, 4, 1]
}
df = pd.DataFrame(data)

# 説明変数と目的変数に分ける
X = df[['X1', 'X2', 'X3']]
y = df['Y']

# 定数項を追加
X = sm.add_constant(X)

# モデルの作成と学習
model = sm.OLS(y, X).fit()

# 結果の表示
print(model.summary())