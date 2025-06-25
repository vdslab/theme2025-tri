# Issues - ゲームデータクラスタリング可視化プロジェクト

## プロジェクト概要

testdata.json のゲームデータを使用して、特徴量によるクラスタリングを行い、D3.js で可視化するプロジェクト

## Issue 一覧

| #   | タイトル                                                | 優先度 | 見積もり | ステータス |
| --- | ------------------------------------------------------- | ------ | -------- | ---------- |
| 01  | [データ処理・前処理](./01-data-preprocessing.md)        | High   | 2-3h     | ✅ 完了    |
| 02  | [クラスタリング実装](./02-clustering-implementation.md) | High   | 3-4h     | ✅ 完了    |
| 03  | [D3.js 可視化実装](./03-d3js-visualization.md)          | High   | 4-5h     | ✅ 完了    |
| 04  | [UI/UX 改善](./04-ui-ux-improvements.md)                | Medium | 2-3h     | 🚧 進行中  |
| 05  | [統合・テスト](./05-integration-testing.md)             | High   | 2-3h     | 未着手     |

## 総見積もり時間

13-18 時間

## 実装順序

1. Issue #01: データ処理・前処理
2. Issue #02: クラスタリング実装
3. Issue #03: D3.js 可視化実装
4. Issue #04: UI/UX 改善
5. Issue #05: 統合・テスト

## 技術スタック

- JavaScript/ES6+
- D3.js (可視化)
- HTML5/CSS3
- Vite (ビルドツール)

## データ仕様

- **入力**: `frontend/data/testdata.json`
- **特徴量**: time, ex_base_hit_cnt, total_score, diff_score, lead_change_cnt
- **正規化**: 0〜1 の範囲に Min-Max スケーリング
