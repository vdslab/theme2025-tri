# theme2025-tri

# 概要

MLB の試合データを用いた盛り上がりの可視化

## ブランチワーク

ロジスティック回帰分析を行うコードと行った後のデータ成型を本ブランチで行う。
### 開発時

- メインブランチ: develop

- 機能バグ起票時: issue を立てる

  - xxxxxxxxxxxxx #17 という風に #num と出るので、num を以後使用する

- 機能追加開発時: develop ブランチから feat/num_xxx_xxx_xxx... (例: feat/17_xxx_xxx_xxx...)ブランチを作成して、develop ブランチにマージリクエストする

- コミット時:
  - 機能追加時: commit -m "#num feat: xxxxxxxxxxxxx" (例: #17 feat: xxxxxxxxxxxxx)
  - バグ改修時: commit -m "#num fix: xxxxxxxxxxxxx" (例: #17 fix: xxxxxxxxxxxxx)