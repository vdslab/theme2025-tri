/**
 * K-means クラスタリング実装
 * Issue #02: クラスタリング実装
 */

/**
 * ユークリッド距離を計算
 * @param {Array} point1 - 点1の座標
 * @param {Array} point2 - 点2の座標
 * @returns {number} - ユークリッド距離
 */
function euclideanDistance(point1, point2) {
  return Math.sqrt(
    point1.reduce((sum, val, i) => sum + Math.pow(val - point2[i], 2), 0),
  );
}

/**
 * データポイントを特徴量ベクトルに変換
 * @param {Array} data - データ
 * @param {Array} features - 使用する特徴量
 * @returns {Array} - 特徴量ベクトルの配列
 */
function dataToVectors(data, features) {
  return data.map((item) => features.map((feature) => item[feature]));
}

/**
 * K-means クラスタリングを実行
 * @param {Array} data - 正規化されたデータ
 * @param {number} k - クラスタ数
 * @param {Array} features - 使用する特徴量
 * @param {number} maxIterations - 最大反復回数
 * @param {number} tolerance - 収束判定の閾値
 * @returns {Object} - { clusters, centroids, iterations, inertia }
 */
export function kmeans(
  data,
  k,
  features,
  maxIterations = 100,
  tolerance = 1e-4,
) {
  const vectors = dataToVectors(data, features);
  const n = vectors.length;
  const d = features.length;

  // 初期重心をランダムに設定
  let centroids = [];
  for (let i = 0; i < k; i++) {
    const centroid = [];
    for (let j = 0; j < d; j++) {
      centroid.push(Math.random());
    }
    centroids.push(centroid);
  }

  let clusters = new Array(n);
  let iterations = 0;
  let converged = false;

  while (iterations < maxIterations && !converged) {
    // 各点を最も近い重心に割り当て
    const newClusters = new Array(n);

    for (let i = 0; i < n; i++) {
      let minDistance = Infinity;
      let closestCentroid = 0;

      for (let j = 0; j < k; j++) {
        const distance = euclideanDistance(vectors[i], centroids[j]);
        if (distance < minDistance) {
          minDistance = distance;
          closestCentroid = j;
        }
      }

      newClusters[i] = closestCentroid;
    }

    // 重心を更新
    const newCentroids = [];
    for (let i = 0; i < k; i++) {
      const clusterPoints = vectors.filter((_, idx) => newClusters[idx] === i);

      if (clusterPoints.length === 0) {
        // 空のクラスタの場合はランダムに重心を設定
        const centroid = [];
        for (let j = 0; j < d; j++) {
          centroid.push(Math.random());
        }
        newCentroids.push(centroid);
      } else {
        // 平均を計算
        const centroid = [];
        for (let j = 0; j < d; j++) {
          const sum = clusterPoints.reduce((acc, point) => acc + point[j], 0);
          centroid.push(sum / clusterPoints.length);
        }
        newCentroids.push(centroid);
      }
    }

    // 収束判定
    const centroidShift = centroids.reduce((maxShift, oldCentroid, i) => {
      const shift = euclideanDistance(oldCentroid, newCentroids[i]);
      return Math.max(maxShift, shift);
    }, 0);

    converged = centroidShift < tolerance;
    clusters = newClusters;
    centroids = newCentroids;
    iterations++;
  }

  // 慣性（Within-cluster sum of squares）を計算
  let inertia = 0;
  for (let i = 0; i < n; i++) {
    const clusterIndex = clusters[i];
    const distance = euclideanDistance(vectors[i], centroids[clusterIndex]);
    inertia += distance * distance;
  }

  return { clusters, centroids, iterations, inertia };
}

/**
 * エルボー法で最適なクラスタ数を決定
 * @param {Array} data - 正規化されたデータ
 * @param {Array} features - 使用する特徴量
 * @param {number} maxK - 試行する最大クラスタ数
 * @returns {Object} - { optimalK, elbowData }
 */
export function findOptimalClusters(data, features, maxK = 10) {
  const elbowData = [];

  for (let k = 1; k <= maxK; k++) {
    const result = kmeans(data, k, features);
    elbowData.push({
      k,
      inertia: result.inertia,
      iterations: result.iterations,
    });
  }

  // エルボーポイントを見つける（簡易版）
  let optimalK = 3; // デフォルト値
  let maxImprovement = 0;

  for (let i = 1; i < elbowData.length - 1; i++) {
    const prevInertia = elbowData[i - 1].inertia;
    const currInertia = elbowData[i].inertia;
    const nextInertia = elbowData[i + 1].inertia;

    const improvement1 = prevInertia - currInertia;
    const improvement2 = currInertia - nextInertia;
    const elbowScore = improvement1 - improvement2;

    if (elbowScore > maxImprovement) {
      maxImprovement = elbowScore;
      optimalK = i + 1;
    }
  }

  return { optimalK, elbowData };
}

/**
 * クラスタリングのメイン関数
 * @param {Array} normalizedData - 正規化されたデータ
 * @param {Array} features - 使用する特徴量
 * @param {number} k - クラスタ数（指定しない場合は自動決定）
 * @returns {Object} - クラスタリング結果
 */
export function performClustering(normalizedData, features, k = null) {
  console.log("🔄 クラスタリングを開始します...");

  // クラスタ数が指定されていない場合は自動決定
  if (k === null) {
    console.log("📊 最適なクラスタ数を決定中...");
    const { optimalK, elbowData } = findOptimalClusters(
      normalizedData,
      features,
    );
    k = optimalK;
    console.log(`✅ 最適なクラスタ数: ${k}`);
  }

  // K-means クラスタリングを実行
  const result = kmeans(normalizedData, k, features);

  // 各データポイントにクラスタラベルを付与
  const clusteredData = normalizedData.map((item, index) => ({
    ...item,
    cluster: result.clusters[index],
  }));

  console.log(`✅ クラスタリング完了: ${result.iterations} 回の反復で収束`);
  console.log(`📈 慣性: ${result.inertia.toFixed(4)}`);

  return {
    clusteredData,
    centroids: result.centroids,
    k,
    iterations: result.iterations,
    inertia: result.inertia,
    features,
  };
}
