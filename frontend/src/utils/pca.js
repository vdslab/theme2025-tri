/**
 * 主成分分析（PCA）実装
 * 5次元特徴量を2次元に次元削減
 */

// 未使用の行列演算関数を削除（linter対応）

/**
 * ベクトルの平均を計算
 * @param {Array} vectors - ベクトルの配列
 * @returns {Array} - 平均ベクトル
 */
function calculateMean(vectors) {
  const dimensions = vectors[0].length;
  const mean = new Array(dimensions).fill(0);

  for (const vector of vectors) {
    for (let i = 0; i < dimensions; i++) {
      mean[i] += vector[i];
    }
  }

  return mean.map((sum) => sum / vectors.length);
}

/**
 * データを中心化（平均を0にする）
 * @param {Array} vectors - ベクトルの配列
 * @param {Array} mean - 平均ベクトル
 * @returns {Array} - 中心化されたベクトル
 */
function centerData(vectors, mean) {
  return vectors.map((vector) => vector.map((value, i) => value - mean[i]));
}

/**
 * 共分散行列を計算
 * @param {Array} centeredData - 中心化されたデータ
 * @returns {Array} - 共分散行列
 */
function calculateCovarianceMatrix(centeredData) {
  const n = centeredData.length;
  const dimensions = centeredData[0].length;
  const covariance = [];

  for (let i = 0; i < dimensions; i++) {
    covariance[i] = [];
    for (let j = 0; j < dimensions; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += centeredData[k][i] * centeredData[k][j];
      }
      covariance[i][j] = sum / (n - 1);
    }
  }

  return covariance;
}

/**
 * 累乗法で最大固有値と固有ベクトルを求める
 * @param {Array} matrix - 正方行列
 * @param {number} maxIterations - 最大反復回数
 * @param {number} tolerance - 収束判定の閾値
 * @returns {Object} - { eigenvalue, eigenvector }
 */
function powerMethod(matrix, maxIterations = 1000, tolerance = 1e-6) {
  const n = matrix.length;
  let vector = new Array(n).fill(1); // 初期ベクトル
  let eigenvalue = 0;

  for (let iter = 0; iter < maxIterations; iter++) {
    // Av を計算
    const newVector = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        newVector[i] += matrix[i][j] * vector[j];
      }
    }

    // ノルムを計算
    const norm = Math.sqrt(newVector.reduce((sum, val) => sum + val * val, 0));

    // 正規化
    const normalizedVector = newVector.map((val) => val / norm);

    // 固有値を計算（レイリー商）
    let numerator = 0;
    let denominator = 0;
    for (let i = 0; i < n; i++) {
      numerator += normalizedVector[i] * newVector[i];
      denominator += normalizedVector[i] * vector[i];
    }
    const newEigenvalue = numerator / denominator;

    // 収束判定
    if (Math.abs(newEigenvalue - eigenvalue) < tolerance) {
      return { eigenvalue: newEigenvalue, eigenvector: normalizedVector };
    }

    eigenvalue = newEigenvalue;
    vector = normalizedVector;
  }

  return { eigenvalue, eigenvector: vector };
}

/**
 * 行列から指定した固有ベクトルの成分を除去（デフレーション）
 * @param {Array} matrix - 元の行列
 * @param {number} eigenvalue - 固有値
 * @param {Array} eigenvector - 固有ベクトル
 * @returns {Array} - デフレーションされた行列
 */
function deflation(matrix, eigenvalue, eigenvector) {
  const n = matrix.length;
  const deflatedMatrix = [];

  for (let i = 0; i < n; i++) {
    deflatedMatrix[i] = [];
    for (let j = 0; j < n; j++) {
      deflatedMatrix[i][j] =
        matrix[i][j] - eigenvalue * eigenvector[i] * eigenvector[j];
    }
  }

  return deflatedMatrix;
}

/**
 * 主成分分析を実行
 * @param {Array} data - 正規化されたデータ（各行がサンプル、各列が特徴量）
 * @param {Array} features - 特徴量名の配列
 * @param {number} components - 取得する主成分の数（デフォルト: 2）
 * @returns {Object} - PCA結果
 */
export function performPCA(data, features, components = 2) {
  console.log("🔄 PCA（主成分分析）を開始します...");

  // データを数値ベクトルに変換
  const vectors = data.map((item) => features.map((feature) => item[feature]));

  console.log(
    `📊 データサイズ: ${vectors.length} サンプル × ${features.length} 特徴量`
  );

  // 1. 平均を計算
  const mean = calculateMean(vectors);
  console.log(
    "平均ベクトル:",
    mean.map((val) => val.toFixed(4))
  );

  // 2. データを中心化
  const centeredData = centerData(vectors, mean);

  // 3. 共分散行列を計算
  const covarianceMatrix = calculateCovarianceMatrix(centeredData);
  console.log("共分散行列を計算しました");

  // 4. 主要な固有値・固有ベクトルを計算
  const principalComponents = [];
  let currentMatrix = covarianceMatrix;

  for (let i = 0; i < components; i++) {
    const { eigenvalue, eigenvector } = powerMethod(currentMatrix);
    principalComponents.push({
      eigenvalue,
      eigenvector,
      varianceRatio:
        eigenvalue /
        principalComponents.reduce(
          (sum, pc) => sum + pc.eigenvalue,
          eigenvalue
        ),
    });

    console.log(`主成分${i + 1}: 固有値=${eigenvalue.toFixed(4)}`);

    // 次の主成分のためにデフレーション
    if (i < components - 1) {
      currentMatrix = deflation(currentMatrix, eigenvalue, eigenvector);
    }
  }

  // 5. データを主成分空間に投影
  const projectedData = centeredData.map((vector) => {
    return principalComponents.map((pc) => {
      return vector.reduce(
        (sum, val, idx) => sum + val * pc.eigenvector[idx],
        0
      );
    });
  });

  // 6. 結果をオリジナルデータと組み合わせ
  const pcaData = data.map((item, index) => ({
    ...item,
    pc1: projectedData[index][0],
    pc2: projectedData[index][1],
  }));

  // 寄与率を計算
  const totalVariance = principalComponents.reduce(
    (sum, pc) => sum + pc.eigenvalue,
    0
  );
  const varianceRatios = principalComponents.map(
    (pc) => pc.eigenvalue / totalVariance
  );

  console.log(`✅ PCA完了`);
  console.log(`第1主成分の寄与率: ${(varianceRatios[0] * 100).toFixed(1)}%`);
  console.log(`第2主成分の寄与率: ${(varianceRatios[1] * 100).toFixed(1)}%`);
  console.log(
    `累積寄与率: ${(
      varianceRatios.slice(0, 2).reduce((sum, ratio) => sum + ratio, 0) * 100
    ).toFixed(1)}%`
  );

  return {
    data: pcaData,
    principalComponents,
    varianceRatios,
    totalVariance,
    mean,
    features,
    projectedData,
  };
}
