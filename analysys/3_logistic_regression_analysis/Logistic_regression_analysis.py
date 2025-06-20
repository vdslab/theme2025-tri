import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris

# データの読み込み（例: CSVファイル）
# df = pd.read_csv('your_data.csv')

# 仮のデータ例
# X = df[['feature1', 'feature2', ...]]
# y = df['target']

# サンプルデータ（削除してご自身のデータに置き換えてください）
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # 2値分類用に変換

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ロジスティック回帰モデルの作成と学習
model = LogisticRegression()
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 結果の評価
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))