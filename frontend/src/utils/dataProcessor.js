/**
 * データ処理・前処理ユーティリティ
 * Issue #01: データ処理・前処理
 */

/**
 * JSON データを読み込む
 * @param {string} filePath - JSONファイルのパス
 * @returns {Promise<Array>} - パースされたデータ
 */
export async function loadJsonData(filePath) {
  try {
    const response = await fetch(filePath);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error loading JSON data:", error);
    throw error;
  }
}

/**
 * gamepk 以外の特徴量を抽出し、date と team 情報も保持
 * @param {Array} rawData - 元データ
 * @returns {Array} - 特徴量データ
 */
export function extractFeatures(rawData) {
  const features = [
    "time",
    "ex_base_hit_cnt",
    "total_score",
    "diff_score",
    "lead_change_cnt",
  ];

  return rawData.map((item) => {
    const featureData = {};

    // gamepk を保持（可視化で使用）
    featureData.gamepk = item.gamepk;

    // 特徴量を抽出
    features.forEach((feature) => {
      featureData[feature] = item[feature];
    });

    // date と team 情報も保持（ツールチップ表示用）
    if (item.date) {
      featureData.date = item.date;
    }

    if (item.team) {
      featureData.team = item.team;
    }

    return featureData;
  });
}

/**
 * Min-Max スケーリングで特徴量を 0〜1 の範囲に正規化
 * @param {Array} data - 特徴量データ
 * @param {Array} features - 正規化する特徴量名のリスト
 * @returns {Object} - { normalizedData, scalingParams }
 */
export function normalizeFeatures(
  data,
  features = [
    "time",
    "ex_base_hit_cnt",
    "total_score",
    "diff_score",
    "lead_change_cnt",
  ]
) {
  // 各特徴量の最小値・最大値を計算
  const scalingParams = {};

  features.forEach((feature) => {
    const values = data.map((item) => item[feature]);
    const min = Math.min(...values);
    const max = Math.max(...values);

    scalingParams[feature] = { min, max, range: max - min };
  });

  // Min-Max スケーリングを適用
  const normalizedData = data.map((item) => {
    const normalizedItem = { ...item };

    features.forEach((feature) => {
      const { min, range } = scalingParams[feature];
      if (range === 0) {
        // 全ての値が同じ場合は 0 に設定
        normalizedItem[feature] = 0;
      } else {
        normalizedItem[feature] = (item[feature] - min) / range;
      }
    });

    return normalizedItem;
  });

  return { normalizedData, scalingParams };
}

/**
 * 正規化を逆変換（デバッグ用）
 * @param {Array} normalizedData - 正規化されたデータ
 * @param {Object} scalingParams - スケーリングパラメータ
 * @param {Array} features - 逆変換する特徴量名のリスト
 * @returns {Array} - 元のスケールに戻されたデータ
 */
export function denormalizeFeatures(
  normalizedData,
  scalingParams,
  features = [
    "time",
    "ex_base_hit_cnt",
    "total_score",
    "diff_score",
    "lead_change_cnt",
  ]
) {
  return normalizedData.map((item) => {
    const denormalizedItem = { ...item };

    features.forEach((feature) => {
      const { min, range } = scalingParams[feature];
      denormalizedItem[feature] = item[feature] * range + min;
    });

    return denormalizedItem;
  });
}

/**
 * データ処理のメイン関数
 * @param {string} filePath - JSONファイルのパス
 * @returns {Promise<Object>} - { originalData, featureData, normalizedData, scalingParams }
 */
export async function processGameData(filePath = "/data/testdata.json") {
  try {
    // 1. JSON データを読み込み
    const originalData = await loadJsonData(filePath);
    console.log(`✅ JSON データを読み込みました: ${originalData.length} 件`);

    // 2. 特徴量を抽出
    const featureData = extractFeatures(originalData);
    console.log("✅ 4 つの特徴量を抽出しました");

    // 3. 特徴量を正規化
    const features = [
      "time",
      "ex_base_hit_cnt",
      "total_score",
      "diff_score",
      "lead_change_cnt",
    ];
    const { normalizedData, scalingParams } = normalizeFeatures(
      featureData,
      features
    );
    console.log("✅ 特徴量を 0〜1 の範囲に正規化しました");

    // 正規化結果の検証
    console.log("正規化パラメータ:", scalingParams);

    return {
      originalData,
      featureData,
      normalizedData,
      scalingParams,
      features,
    };
  } catch (error) {
    console.error("❌ データ処理エラー:", error);
    throw error;
  }
}
