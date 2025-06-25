import React from "react";
import "./App.css";
import ClusterVisualization from "./components/ClusterVisualization";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>ゲームデータ クラスタリング可視化</h1>
        <p>野球ゲームデータの特徴量によるクラスタリング分析</p>
      </header>

      <main>
        <ClusterVisualization />
      </main>

      <footer
        style={{
          textAlign: "center",
          padding: "20px",
          borderTop: "1px solid #eee",
          marginTop: "40px",
          color: "#666",
        }}
      >
        <p>D3.js を使用したインタラクティブ可視化</p>
      </footer>
    </div>
  );
}

export default App;
