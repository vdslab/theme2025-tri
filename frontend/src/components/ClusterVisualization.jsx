import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { processGameData } from "../utils/dataProcessor";
import { performClustering, findOptimalClusters } from "../utils/kmeans";
import { performPCA } from "../utils/pca";

/**
 * クラスタリング可視化コンポーネント
 * Issue #03: D3.js 可視化実装（PCA対応）
 * Issue #04: UI/UX改善（手動クラスタ数指定）
 */
const ClusterVisualization = () => {
  const svgRef = useRef();
  const elbowChartRef = useRef();
  const [data, setData] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [elbowData, setElbowData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [clusterCount, setClusterCount] = useState(null); // 手動指定されたクラスタ数
  const [isAutoMode, setIsAutoMode] = useState(true); // 自動/手動モード切り替え
  const [updating, setUpdating] = useState(false);

  // 初期データの読み込み
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setLoading(true);

        // データ処理
        const processedData = await processGameData(
          "/data/2025-03-16-2025-06-16.json"
        );

        // エルボー法でクラスタ数候補を分析
        console.log("📊 エルボー分析を実行中...");
        const { optimalK, elbowData: elbowResult } = findOptimalClusters(
          processedData.normalizedData,
          processedData.features,
          8 // 最大8クラスタまで試行
        );

        setProcessedData(processedData);
        setElbowData(elbowResult);
        setClusterCount(optimalK);

        console.log(`✅ 推奨クラスタ数: ${optimalK}`);
      } catch (err) {
        console.error("データ読み込みエラー:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();
  }, []);

  // クラスタリング実行（クラスタ数変更時）
  useEffect(() => {
    if (!processedData || !clusterCount) return;

    const performClusteringAndPCA = async () => {
      try {
        setUpdating(true);

        // 指定されたクラスタ数でクラスタリング実行
        const clusteringResult = performClustering(
          processedData.normalizedData,
          processedData.features,
          clusterCount // 手動指定されたクラスタ数を使用
        );

        // PCAで次元削減
        const pcaResult = performPCA(
          clusteringResult.clusteredData,
          processedData.features,
          2
        );

        setData({
          ...processedData,
          ...clusteringResult,
          pca: pcaResult,
        });
      } catch (err) {
        console.error("クラスタリングエラー:", err);
        setError(err.message);
      } finally {
        setUpdating(false);
      }
    };

    performClusteringAndPCA();
  }, [processedData, clusterCount]);

  // エルボー曲線の描画
  useEffect(() => {
    if (!elbowData) return;

    const drawElbowChart = () => {
      const svg = d3.select(elbowChartRef.current);
      svg.selectAll("*").remove();

      const margin = { top: 20, right: 20, bottom: 40, left: 50 };
      const width = 300 - margin.left - margin.right;
      const height = 200 - margin.top - margin.bottom;

      const g = svg
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const xScale = d3
        .scaleLinear()
        .domain(d3.extent(elbowData, (d) => d.k))
        .range([0, width]);

      const yScale = d3
        .scaleLinear()
        .domain(d3.extent(elbowData, (d) => d.inertia))
        .range([height, 0]);

      // 軸
      g.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(
          d3
            .axisBottom(xScale)
            .ticks(elbowData.length)
            .tickFormat(d3.format("d"))
        );

      g.append("g").call(d3.axisLeft(yScale).ticks(5));

      // ラベル
      g.append("text")
        .attr("x", width / 2)
        .attr("y", height + 35)
        .attr("text-anchor", "middle")
        .style("font-size", "12px")
        .text("クラスタ数");

      g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -35)
        .attr("text-anchor", "middle")
        .style("font-size", "12px")
        .text("慣性");

      // 線グラフ
      const line = d3
        .line()
        .x((d) => xScale(d.k))
        .y((d) => yScale(d.inertia))
        .curve(d3.curveMonotoneX);

      g.append("path")
        .datum(elbowData)
        .attr("fill", "none")
        .attr("stroke", "#667eea")
        .attr("stroke-width", 2)
        .attr("d", line);

      // ポイント
      g.selectAll(".dot")
        .data(elbowData)
        .join("circle")
        .attr("class", "dot")
        .attr("cx", (d) => xScale(d.k))
        .attr("cy", (d) => yScale(d.inertia))
        .attr("r", (d) => (d.k === clusterCount ? 6 : 4))
        .attr("fill", (d) => (d.k === clusterCount ? "#ff6b6b" : "#667eea"))
        .attr("stroke", "white")
        .attr("stroke-width", 2);
    };

    drawElbowChart();
  }, [elbowData, clusterCount]);

  // D3.js 可視化（PCA版）
  useEffect(() => {
    if (!data) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // 既存の要素をクリア

    const margin = { top: 60, right: 140, bottom: 80, left: 80 };
    const width = 900 - margin.left - margin.right;
    const height = 700 - margin.top - margin.bottom;

    // メインのグループ要素を作成
    const g = svg
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // カラースケール（クラスタごとの色）
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // PCAデータのスケール設定
    const pcaData = data.pca.data;
    const xExtent = d3.extent(pcaData, (d) => d.pc1);
    const yExtent = d3.extent(pcaData, (d) => d.pc2);

    // マージンを追加
    const xRange = xExtent[1] - xExtent[0];
    const yRange = yExtent[1] - yExtent[0];
    const xMargin = xRange * 0.1;
    const yMargin = yRange * 0.1;

    const xScale = d3
      .scaleLinear()
      .domain([xExtent[0] - xMargin, xExtent[1] + xMargin])
      .range([0, width]);

    const yScale = d3
      .scaleLinear()
      .domain([yExtent[0] - yMargin, yExtent[1] + yMargin])
      .range([height, 0]);

    // 軸を作成
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat((d) => d.toFixed(2)))
      .append("text")
      .attr("x", width / 2)
      .attr("y", 50)
      .attr("fill", "black")
      .style("text-anchor", "middle")
      .style("font-size", "14px")
      .text(
        `第1主成分 (寄与率: ${(data.pca.varianceRatios[0] * 100).toFixed(1)}%)`
      );

    g.append("g")
      .call(d3.axisLeft(yScale).tickFormat((d) => d.toFixed(2)))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -50)
      .attr("x", -height / 2)
      .attr("fill", "black")
      .style("text-anchor", "middle")
      .style("font-size", "14px")
      .text(
        `第2主成分 (寄与率: ${(data.pca.varianceRatios[1] * 100).toFixed(1)}%)`
      );

    // グリッドライン
    g.append("g")
      .attr("class", "grid")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickSize(-height).tickFormat(""))
      .style("stroke-dasharray", "3,3")
      .style("opacity", 0.3);

    g.append("g")
      .attr("class", "grid")
      .call(d3.axisLeft(yScale).tickSize(-width).tickFormat(""))
      .style("stroke-dasharray", "3,3")
      .style("opacity", 0.3);

    // ツールチップ用のdiv
    const tooltip = d3
      .select("body")
      .selectAll(".tooltip")
      .data([null])
      .join("div")
      .attr("class", "tooltip")
      .style("position", "absolute")
      .style("padding", "12px")
      .style("background", "rgba(0, 0, 0, 0.9)")
      .style("color", "white")
      .style("border-radius", "8px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("max-width", "300px")
      .style("box-shadow", "0 4px 12px rgba(0,0,0,0.3)");

    // データポイント（ノード）を描画
    g.selectAll(".node")
      .data(pcaData)
      .join("circle")
      .attr("class", "node")
      .attr("cx", (d) => xScale(d.pc1))
      .attr("cy", (d) => yScale(d.pc2))
      .attr("r", 7)
      .attr("fill", (d) => colorScale(d.cluster))
      .attr("stroke", "white")
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .style("opacity", 0.8)
      .on("mouseover", (event, d) => {
        // ノードをハイライト
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr("r", 10)
          .style("opacity", 1);

        // ツールチップ表示
        tooltip
          .style("opacity", 1)
          .html(
            `
            <div style="border-bottom: 1px solid #555; padding-bottom: 8px; margin-bottom: 8px;">
              <strong style="color: #4FC3F7;">Game PK: ${d.gamepk}</strong><br/>
              <strong style="color: ${colorScale(d.cluster)};">クラスタ ${
              d.cluster
            }</strong>
            </div>
            ${
              d.date
                ? `<div style="margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #555;">
              <strong style="color: #FFD54F;">📅 ${d.date}</strong><br/>
              <strong style="color: #81C784;">${
                d.team ? `${d.team.away} vs ${d.team.home}` : ""
              }</strong>
            </div>`
                : ""
            }
            <div style="line-height: 1.4;">
              <strong>元の特徴量:</strong><br/>
              • 試合時間: ${(
                d.time * data.scalingParams.time.range +
                data.scalingParams.time.min
              ).toFixed(1)}分<br/>
              • エキストラベースヒット: ${Math.round(
                d.ex_base_hit_cnt * data.scalingParams.ex_base_hit_cnt.range +
                  data.scalingParams.ex_base_hit_cnt.min
              )}回<br/>
              • 総得点: ${Math.round(
                d.total_score * data.scalingParams.total_score.range +
                  data.scalingParams.total_score.min
              )}点<br/>
              • 得点差: ${Math.round(
                d.diff_score * data.scalingParams.diff_score.range +
                  data.scalingParams.diff_score.min
              )}点<br/>
              • リードチェンジ: ${Math.round(
                d.lead_change_cnt * data.scalingParams.lead_change_cnt.range +
                  data.scalingParams.lead_change_cnt.min
              )}回
            </div>
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #555; font-size: 11px; color: #ccc;">
              PC1: ${d.pc1.toFixed(3)}, PC2: ${d.pc2.toFixed(3)}
            </div>
          `
          )
          .style("left", event.pageX + 15 + "px")
          .style("top", event.pageY - 10 + "px");
      })
      .on("mouseout", (event) => {
        // ノードを元に戻す
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr("r", 7)
          .style("opacity", 0.8);

        tooltip.style("opacity", 0);
      })
      .on("click", (event, d) => {
        const textToCopy = d.gamepk;
        navigator.clipboard
          .writeText(textToCopy)
          .then(function () {
            alert("テキストをコピーしました！ copy gamepk = " + textToCopy);
          })
          .catch(function (error) {
            alert("テキストのコピーに失敗しました");
            console.log(error);
          });
      });

    // 凡例を作成
    const legend = svg
      .append("g")
      .attr(
        "transform",
        `translate(${width + margin.left + 20}, ${margin.top})`
      );

    const clusters = [...new Set(pcaData.map((d) => d.cluster))].sort(
      (a, b) => a - b
    );

    legend
      .append("text")
      .attr("x", 0)
      .attr("y", -10)
      .text("クラスタ")
      .style("font-size", "16px")
      .style("font-weight", "bold");

    clusters.forEach((cluster, i) => {
      const legendItem = legend
        .append("g")
        .attr("transform", `translate(0, ${i * 30 + 15})`);

      legendItem
        .append("circle")
        .attr("r", 7)
        .attr("fill", colorScale(cluster))
        .attr("stroke", "white")
        .attr("stroke-width", 2);

      const clusterSize = pcaData.filter((d) => d.cluster === cluster).length;
      legendItem
        .append("text")
        .attr("x", 20)
        .attr("y", 5)
        .text(`クラスタ ${cluster} (${clusterSize}件)`)
        .style("font-size", "14px")
        .style("alignment-baseline", "middle");
    });

    // 寄与率情報
    const varianceInfo = legend
      .append("g")
      .attr("transform", `translate(0, ${clusters.length * 30 + 40})`);

    varianceInfo
      .append("text")
      .attr("x", 0)
      .attr("y", 0)
      .text("寄与率")
      .style("font-size", "16px")
      .style("font-weight", "bold");

    const cumulativeVariance = data.pca.varianceRatios
      .slice(0, 2)
      .reduce((sum, ratio) => sum + ratio, 0);

    varianceInfo
      .append("text")
      .attr("x", 0)
      .attr("y", 25)
      .text(`累積: ${(cumulativeVariance * 100).toFixed(1)}%`)
      .style("font-size", "14px")
      .style("fill", "#666");

    // タイトル
    svg
      .append("text")
      .attr("x", (width + margin.left + margin.right) / 2)
      .attr("y", 30)
      .attr("text-anchor", "middle")
      .style("font-size", "18px")
      .style("font-weight", "bold")
      .text("ゲームデータクラスタリング可視化（PCA）");

    svg
      .append("text")
      .attr("x", (width + margin.left + margin.right) / 2)
      .attr("y", 50)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("fill", "#666")
      .text("5次元特徴量を主成分分析で2次元に次元削減");
  }, [data]);

  // 自動モード切り替え時の処理
  const handleModeChange = (auto) => {
    setIsAutoMode(auto);
    if (auto && elbowData) {
      // 自動モードの場合、推奨クラスタ数を使用
      const optimalK = elbowData.reduce(
        (optimal, current, index) => {
          if (index === 0) return optimal;
          const prev = elbowData[index - 1];
          const improvement = prev.inertia - current.inertia;
          return improvement > optimal.improvement
            ? { k: current.k, improvement }
            : optimal;
        },
        { k: 3, improvement: 0 }
      ).k;

      setClusterCount(optimalK);
    }
  };

  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "400px",
        }}
      >
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: "18px", marginBottom: "10px" }}>
            🔄 データを処理中...
          </div>
          <div style={{ fontSize: "14px", color: "#666" }}>
            データ読み込み → 特徴量抽出 → 正規化 → エルボー分析
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "400px",
          color: "red",
        }}
      >
        <div>エラー: {error}</div>
      </div>
    );
  }

  return (
    <div style={{ padding: "20px" }}>
      {/* クラスタ数制御パネル */}
      <div
        className="cluster-control-panel"
        style={{
          background: "white",
          padding: "20px",
          borderRadius: "8px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
          marginBottom: "20px",
          display: "grid",
          gridTemplateColumns: "1fr 300px",
          gap: "20px",
          alignItems: "center",
        }}
      >
        <div>
          <h3 style={{ margin: "0 0 15px 0" }}>🎛️ クラスタ数設定</h3>

          {/* 自動/手動切り替え */}
          <div style={{ marginBottom: "15px" }}>
            <label style={{ marginRight: "15px" }}>
              <input
                type="radio"
                name="mode"
                checked={isAutoMode}
                onChange={() => handleModeChange(true)}
                style={{ marginRight: "5px" }}
              />
              自動決定（エルボー法）
            </label>
            <label>
              <input
                type="radio"
                name="mode"
                checked={!isAutoMode}
                onChange={() => handleModeChange(false)}
                style={{ marginRight: "5px" }}
              />
              手動指定
            </label>
          </div>

          {/* クラスタ数スライダー */}
          <div style={{ marginBottom: "10px" }}>
            <label
              style={{
                display: "block",
                marginBottom: "5px",
                fontWeight: "bold",
              }}
            >
              クラスタ数: {clusterCount} {updating && "⟳"}
            </label>
            <input
              type="range"
              min="2"
              max="8"
              value={clusterCount || 3}
              onChange={(e) => {
                setIsAutoMode(false);
                setClusterCount(parseInt(e.target.value));
              }}
              disabled={isAutoMode}
              style={{
                width: "100%",
                opacity: isAutoMode ? 0.5 : 1,
                cursor: isAutoMode ? "not-allowed" : "pointer",
              }}
            />
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "12px",
                color: "#666",
              }}
            >
              <span>2</span>
              <span>3</span>
              <span>4</span>
              <span>5</span>
              <span>6</span>
              <span>7</span>
              <span>8</span>
            </div>
          </div>

          {isAutoMode && elbowData && (
            <div style={{ fontSize: "14px", color: "#666" }}>
              💡 エルボー法により最適なクラスタ数が自動選択されています
            </div>
          )}
        </div>

        {/* エルボー曲線 */}
        <div>
          <h4 style={{ margin: "0 0 10px 0", fontSize: "14px" }}>
            📈 エルボー曲線
          </h4>
          <svg
            ref={elbowChartRef}
            style={{ border: "1px solid #eee", borderRadius: "4px" }}
          ></svg>
          <div style={{ fontSize: "12px", color: "#666", marginTop: "5px" }}>
            赤点: 現在のクラスタ数
          </div>
        </div>
      </div>

      {data && (
        <div style={{ marginBottom: "20px" }}>
          <div className="stats-info">
            <h3>分析結果</h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                gap: "15px",
              }}
            >
              <div>
                <strong>データ数:</strong> {data.clusteredData.length} 件<br />
                <strong>クラスタ数:</strong> {data.k}{" "}
                {!isAutoMode && "(手動指定)"}
                <br />
                <strong>反復回数:</strong> {data.iterations}
              </div>
              <div>
                <strong>第1主成分:</strong>{" "}
                {(data.pca.varianceRatios[0] * 100).toFixed(1)}%<br />
                <strong>第2主成分:</strong>{" "}
                {(data.pca.varianceRatios[1] * 100).toFixed(1)}%<br />
                <strong>累積寄与率:</strong>{" "}
                {(
                  (data.pca.varianceRatios[0] + data.pca.varianceRatios[1]) *
                  100
                ).toFixed(1)}
                %
              </div>
              <div>
                <strong>慣性:</strong> {data.inertia.toFixed(4)}
                <br />
                <strong>使用特徴量:</strong> {data.features.length}次元
              </div>
            </div>
          </div>
        </div>
      )}

      <div style={{ textAlign: "center", marginBottom: "20px" }}>
        <svg ref={svgRef}></svg>
      </div>

      {data && (
        <div className="feature-info">
          <h4>📊 使用特徴量（5次元）</h4>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
              gap: "10px",
              fontSize: "14px",
            }}
          >
            <div>
              <strong>time:</strong> 試合時間（分）
            </div>
            <div>
              <strong>ex_base_hit_cnt:</strong> エキストラベースヒット数
            </div>
            <div>
              <strong>total_score:</strong> 総得点
            </div>
            <div>
              <strong>diff_score:</strong> 得点差
            </div>
            <div>
              <strong>lead_change_cnt:</strong> リードチェンジ回数
            </div>
          </div>
          <p style={{ marginTop: "15px", fontSize: "14px", color: "#666" }}>
            <strong>💡 操作:</strong> ノードにマウスオーバーで詳細情報を表示 |
            クラスタ数スライダーでリアルタイム変更 |<strong> 💡 解釈:</strong>{" "}
            近くにあるゲームは特徴量が似ており、同じクラスタに分類される傾向があります
          </p>
        </div>
      )}
    </div>
  );
};

export default ClusterVisualization;
