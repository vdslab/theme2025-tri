# stacking_optuna.py
import json
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# scikit-learn / models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss, f1_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.decomposition import PCA
from sklearn.base import clone

# try import optional libs
try:
    import lightgbm as lgb
except Exception as e:
    lgb = None
    print("lightgbm not available:", e)

try:
    from catboost import CatBoostClassifier
except Exception as e:
    CatBoostClassifier = None
    print("catboost not available:", e)

try:
    import optuna
except Exception as e:
    raise ImportError("optuna is required. Install via `pip install optuna`") from e

try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False

# --- 対象試合リスト (あなたのものをそのまま) ---
pk_list = [
    778199, 777579, 777863, 777940, 777571,
    777988, 778062, 778434, 777701, 778444,
    777649, 778220, 778406, 778544, 777726,
    778285, 778262, 778163, 777505,
]

# -----------------------
# ユーティリティ関数群
# -----------------------
def extract_all_feature_paths(play):
    paths = []
    def recurse(d, path):
        for k, v in d.items():
            if isinstance(v, dict):
                if all(isinstance(val, bool) for val in v.values()):
                    paths.append(path + [k])
                recurse(v, path + [k])
    recurse(play, [])
    return paths

def build_feature_df_with_context(data, group_paths, window=1):
    dfs = []
    for play in data:
        combined = {}
        for path in group_paths:
            d = play
            for key in path:
                d = d.get(key, {})
            if isinstance(d, dict):
                for k, v in d.items():
                    combined[".".join(path + [k])] = v
        dfs.append(combined)
    df_main = pd.DataFrame(dfs).fillna(False).astype(int)
    frames = []
    for offset in range(-window, 1):
        df_shifted = df_main.shift(offset)
        label = "cur" if offset == 0 else ("prev" + str(abs(offset)))
        df_shifted.columns = [f"{label}.{col}" for col in df_shifted.columns]
        frames.append(df_shifted)
    return pd.concat(frames, axis=1)

def oof_preds_groupkfold(model_ctor, params, X_train, y_train, groups, X_test, group_kfold_splits=5):
    """
    model_ctor: callable (class), e.g., RandomForestClassifier, lgb.LGBMClassifier, CatBoostClassifier
    params: dict to pass to constructor
    returns: oof_train_probs (n_train,), test_mean_probs (n_test,)
    """
    gkf = GroupKFold(n_splits=group_kfold_splits)
    n_train = len(X_train)
    n_test = len(X_test)
    oof_train = np.zeros(n_train)
    test_preds = np.zeros((group_kfold_splits, n_test))
    fold_i = 0
    for train_idx, val_idx in gkf.split(np.arange(n_train), y_train, groups):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        # instantiate model
        m = model_ctor(**params)
        # CatBoost may warn; set verbose=0 if available
        try:
            m.fit(X_tr, y_tr)
        except TypeError:
            # fallback: try keras style? unlikely. re-raise
            m.fit(X_tr, y_tr)

        oof_train[val_idx] = m.predict_proba(X_val)[:, 1]
        test_preds[fold_i, :] = m.predict_proba(X_test)[:, 1]
        fold_i += 1
    test_mean = test_preds.mean(axis=0)
    return oof_train, test_mean

def compute_aic_from_probs(y_true, probs, k):
    # log_likelihood computed via log_loss with normalize=False yields negative log-likelihood
    ll = -log_loss(y_true, np.vstack([1-probs, probs]).T, normalize=False)
    AIC = 2 * k - 2 * ll
    return AIC

# -----------------------
# Optuna objectives
# -----------------------
def make_rf_objective(X, y, groups, n_splits=5, random_state=42):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 800, step=50)
        max_depth = trial.suggest_categorical("max_depth", [10, 15, 20, None])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7])
        cw_choice = trial.suggest_categorical("class_weight_choice", ["balanced_subsample", "balanced", "dict_2", "dict_3"])
        if cw_choice == "dict_2":
            class_weight = {0:1, 1:2}
        elif cw_choice == "dict_3":
            class_weight = {0:1, 1:3}
        else:
            class_weight = cw_choice

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": -1,
        }

        # GroupKFold CV with F1
        gkf = GroupKFold(n_splits=n_splits)
        f1s = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            X_tr = X.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            m = RandomForestClassifier(**params)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            f1s.append(f1_score(y_val, preds, zero_division=0))
        return float(np.mean(f1s))
    return objective

def make_lgb_objective(X, y, groups, n_splits=5, random_state=42):
    if lgb is None:
        raise RuntimeError("lightgbm is required for LGB objective")
    def objective(trial):
        num_leaves = trial.suggest_categorical("num_leaves", [31, 63, 127])
        max_depth = trial.suggest_categorical("max_depth", [-1, 10, 20])
        learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05])
        n_estimators = trial.suggest_categorical("n_estimators", [200, 500])
        subsample = trial.suggest_categorical("subsample", [0.8, 1.0])
        colsample = trial.suggest_categorical("colsample_bytree", [0.8, 1.0])
        cw_choice = trial.suggest_categorical("class_weight_choice", ["none", "dict_2"])
        class_weight = None if cw_choice == "none" else {0:1,1:2}

        params = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": -1,
        }

        gkf = GroupKFold(n_splits=n_splits)
        f1s = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            X_tr = X.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            f1s.append(f1_score(y_val, preds, zero_division=0))
        return float(np.mean(f1s))
    return objective

def make_cat_objective(X, y, groups, n_splits=5, random_state=42):
    if CatBoostClassifier is None:
        raise RuntimeError("catboost is required for CatBoost objective")
    def objective(trial):
        depth = trial.suggest_categorical("depth", [6, 8, 10])
        iterations = trial.suggest_categorical("iterations", [300, 500, 1000])
        learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.03])
        l2_leaf_reg = trial.suggest_categorical("l2_leaf_reg", [1, 3, 5])
        weight_pos = trial.suggest_categorical("class_weight_pos", [1.5, 2.0, 3.0])

        params = {
            "depth": depth,
            "iterations": iterations,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "random_seed": random_state,
            "verbose": 0,
        }

        # CatBoost expects class_weights as list or dict; we pass as list [w0, w1]
        class_weights = [1.0, weight_pos]

        gkf = GroupKFold(n_splits=n_splits)
        f1s = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            X_tr = X.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            m = CatBoostClassifier(**params, class_weights=class_weights)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            f1s.append(f1_score(y_val, preds, zero_division=0))
        return float(np.mean(f1s))
    return objective

def make_meta_objective(meta_X, meta_y, n_splits=5, random_state=42):
    if lgb is None:
        raise RuntimeError("lightgbm is required for meta objective")
    def objective(trial):
        num_leaves = trial.suggest_categorical("num_leaves", [7, 15, 31, 63])
        max_depth = trial.suggest_categorical("max_depth", [-1, 5, 10])
        learning_rate = trial.suggest_categorical("learning_rate", [0.005, 0.01, 0.05])
        n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 200])
        class_weight_choice = trial.suggest_categorical("class_weight_choice", ["none", "dict_2"])
        class_weight = None if class_weight_choice == "none" else {0:1,1:2}
        params = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": -1,
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        f1s = []
        for tr_idx, val_idx in kf.split(meta_X):
            X_tr = meta_X[tr_idx]
            y_tr = meta_y.iloc[tr_idx]
            X_val = meta_X[val_idx]
            y_val = meta_y.iloc[val_idx]
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr, y_tr)
            probs = m.predict_proba(X_val)[:, 1]
            preds = (probs > 0.5).astype(int)
            f1s.append(f1_score(y_val, preds, zero_division=0))
        return float(np.mean(f1s))
    return objective

# -----------------------
# フルパイプライン
# -----------------------
def run_pipeline(
    windows=[1,2,3],
    TOP_N_FEATURES=50,
    apply_pca_threshold=200,
    n_trials_base=40,
    n_trials_meta=40,
    random_state=42,
    save_shap=True
):
    """
    windows: list of window sizes to run
    TOP_N_FEATURES: not directly used for meta, left for future use
    apply_pca_threshold: if num features > threshold, apply PCA
    n_trials_base: optuna trials for each base model
    n_trials_meta: optuna trials for meta model
    """
    results = []

    for window in windows:
        print("\n" + "="*80)
        print(f"=== Window {window} ===")
        print("="*80)

        # annotation path - change if necessary
        annotation_path = "/content/drive/MyDrive/VDSLab/2025/data/annotation_data/cluster_3.csv"
        annotation_df = pd.read_csv(annotation_path)

        # gather data (like in your previous code)
        all_features_list = []
        all_labels_list = []
        for gamepk in pk_list:
            gamepk_str = str(gamepk)
            molded_path = f"/content/drive/MyDrive/VDSLab/2025/data/molded_data/{gamepk_str}_test2_molded_data.json"
            try:
                with open(molded_path, encoding="utf-8") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"[{gamepk_str}] molded_data not found, skipping")
                continue
            all_plays = [play for minute in data["minutes"].values() for play in minute]
            if not all_plays:
                continue
            if gamepk_str not in annotation_df.columns:
                print(f"[{gamepk_str}] annotation column not found, skipping")
                continue

            group_paths = extract_all_feature_paths(all_plays[0])
            features_df = build_feature_df_with_context(all_plays, group_paths, window=window)
            labels_series = annotation_df.set_index("gamePK")[gamepk_str]
            combined = features_df.copy()
            combined["label"] = labels_series
            combined.dropna(inplace=True)
            if combined.empty:
                continue
            y_final = combined.pop("label").astype(int)
            X_final = combined
            X_final["gamepk"] = gamepk
            all_features_list.append(X_final)
            all_labels_list.append(y_final)

        if not all_features_list:
            print("No processable data for window", window)
            continue

        X_all = pd.concat(all_features_list)
        y_all = pd.concat(all_labels_list)

        # split
        split_point = int(len(pk_list) * 0.75)
        train_pks = set(pk_list[:split_point])
        test_pks = set(pk_list[split_point:])
        train_mask = X_all["gamepk"].isin(train_pks)
        test_mask = X_all["gamepk"].isin(test_pks)

        groups = X_all[train_mask]["gamepk"]
        X_train = X_all[train_mask].drop(columns=["gamepk"])
        y_train = y_all[train_mask].reset_index(drop=True)
        X_test = X_all[test_mask].drop(columns=["gamepk"])
        y_test = y_all[test_mask].reset_index(drop=True)

        if X_train.empty or X_test.empty:
            print("Train or test empty - skip")
            continue

        # optional PCA
        pca_used = False
        if X_train.shape[1] > apply_pca_threshold:
            n_comp = min(100, X_train.shape[1])
            pca = PCA(n_components=n_comp, random_state=random_state)
            X_train = pd.DataFrame(pca.fit_transform(X_train), index=X_train.index)
            X_test = pd.DataFrame(pca.transform(X_test), index=X_test.index)
            pca_used = True

        # ----- RandomForest Optuna -----
        print("-- Optimize RandomForest (Optuna)")
        rf_study = optuna.create_study(direction="maximize", study_name=f"rf_win{window}")
        rf_objective = make_rf_objective(X_train.reset_index(drop=True), y_train.reset_index(drop=True), groups.reset_index(drop=True), n_splits=5, random_state=random_state)
        rf_study.optimize(rf_objective, n_trials=n_trials_base, show_progress_bar=True)
        rf_best_params = rf_study.best_trial.params
        # convert rf_best_params to actual constructor params
        # map class_weight_choice
        cwchoice = rf_best_params.get("class_weight_choice")
        if cwchoice == "dict_2":
            rf_class_weight = {0:1,1:2}
        elif cwchoice == "dict_3":
            rf_class_weight = {0:1,1:3}
        else:
            rf_class_weight = cwchoice
        rf_params_for_model = {
            "n_estimators": rf_best_params["n_estimators"],
            "max_depth": rf_best_params["max_depth"],
            "min_samples_split": rf_best_params["min_samples_split"],
            "min_samples_leaf": rf_best_params["min_samples_leaf"],
            "max_features": rf_best_params["max_features"],
            "class_weight": rf_class_weight,
            "random_state": random_state,
            "n_jobs": -1,
        }

        # produce OOF
        rf_oof_train, rf_oof_test = oof_preds_groupkfold(RandomForestClassifier, rf_params_for_model, X_train.reset_index(drop=True), y_train.reset_index(drop=True), groups.reset_index(drop=True), X_test.reset_index(drop=True), group_kfold_splits=5)
        rf_full_model, rf_report, rf_test_prob = None, None, None
        try:
            rf_full_model, rf_report, rf_test_prob = fit_full_and_eval(RandomForestClassifier, rf_params_for_model, X_train, y_train, X_test, y_test)
        except Exception as e:
            print("RF full fit failed:", e)

        # ----- LightGBM Optuna -----
        if lgb is None:
            print("LightGBM not installed -> skipping LGB")
            lgb_best_params = None
            lgb_oof_train = np.zeros_like(rf_oof_train)
            lgb_oof_test = np.zeros_like(rf_oof_test)
            lgb_full_model = None
            lgb_report = {"興奮 (1)": {"f1-score": np.nan}}
        else:
            print("-- Optimize LightGBM (Optuna)")
            lgb_study = optuna.create_study(direction="maximize", study_name=f"lgb_win{window}")
            lgb_objective = make_lgb_objective(X_train.reset_index(drop=True), y_train.reset_index(drop=True), groups.reset_index(drop=True), n_splits=5, random_state=random_state)
            lgb_study.optimize(lgb_objective, n_trials=n_trials_base, show_progress_bar=True)
            lgb_best_params = lgb_study.best_trial.params
            # translate to constructor params
            lgb_params_for_model = {
                "num_leaves": lgb_best_params["num_leaves"],
                "max_depth": lgb_best_params["max_depth"],
                "learning_rate": lgb_best_params["learning_rate"],
                "n_estimators": lgb_best_params["n_estimators"],
                "subsample": lgb_best_params["subsample"],
                "colsample_bytree": lgb_best_params["colsample_bytree"],
                "class_weight": None if lgb_best_params["class_weight_choice"]=="none" else {0:1,1:2},
                "random_state": random_state,
                "n_jobs": -1,
            }
            lgb_oof_train, lgb_oof_test = oof_preds_groupkfold(lgb.LGBMClassifier, lgb_params_for_model, X_train.reset_index(drop=True), y_train.reset_index(drop=True), groups.reset_index(drop=True), X_test.reset_index(drop=True), group_kfold_splits=5)
            try:
                lgb_full_model, lgb_report, lgb_test_prob = fit_full_and_eval(lgb.LGBMClassifier, lgb_params_for_model, X_train, y_train, X_test, y_test)
            except Exception as e:
                print("LGB full fit failed:", e)
                lgb_report = {"興奮 (1)": {"f1-score": np.nan}}

        # ----- CatBoost Optuna -----
        if CatBoostClassifier is None:
            print("CatBoost not installed -> skipping CatBoost")
            cat_best_params = None
            cat_oof_train = np.zeros_like(rf_oof_train)
            cat_oof_test = np.zeros_like(rf_oof_test)
            cat_full_model = None
            cat_report = {"興奮 (1)": {"f1-score": np.nan}}
        else:
            print("-- Optimize CatBoost (Optuna)")
            cat_study = optuna.create_study(direction="maximize", study_name=f"cat_win{window}")
            cat_objective = make_cat_objective(X_train.reset_index(drop=True), y_train.reset_index(drop=True), groups.reset_index(drop=True), n_splits=5, random_state=random_state)
            cat_study.optimize(cat_objective, n_trials=n_trials_base, show_progress_bar=True)
            cat_best_params = cat_study.best_trial.params
            # convert to constructor params
            cat_params_for_model = {
                "depth": cat_best_params["depth"],
                "iterations": cat_best_params["iterations"],
                "learning_rate": cat_best_params["learning_rate"],
                "l2_leaf_reg": cat_best_params["l2_leaf_reg"],
                "random_seed": random_state,
                "verbose": 0,
            }
            # choose class_weights list
            class_weights = [1.0, float(cat_best_params.get("class_weight_pos", 2.0))]
            cat_oof_train, cat_oof_test = oof_preds_groupkfold(CatBoostClassifier, {**cat_params_for_model, "class_weights": class_weights}, X_train.reset_index(drop=True), y_train.reset_index(drop=True), groups.reset_index(drop=True), X_test.reset_index(drop=True), group_kfold_splits=5)
            try:
                cat_full_model, cat_report, cat_test_prob = fit_full_and_eval(CatBoostClassifier, {**cat_params_for_model, "class_weights": class_weights}, X_train, y_train, X_test, y_test)
            except Exception as e:
                print("CatBoost full fit failed:", e)
                cat_report = {"興奮 (1)": {"f1-score": np.nan}}

        # ---------------------------
        # Build meta features & optimize meta model
        # ---------------------------
        meta_X_train = np.vstack([rf_oof_train, lgb_oof_train, cat_oof_train]).T
        meta_X_test = np.vstack([rf_oof_test, lgb_oof_test, cat_oof_test]).T
        meta_y_train = y_train.reset_index(drop=True)

        print("-- Optimize Meta LightGBM (Optuna)")
        meta_study = optuna.create_study(direction="maximize", study_name=f"meta_win{window}")
        meta_objective = make_meta_objective(meta_X_train, meta_y_train, n_splits=5, random_state=random_state)
        meta_study.optimize(meta_objective, n_trials=n_trials_meta, show_progress_bar=True)
        meta_best_params = meta_study.best_trial.params
        # convert to params
        meta_params_for_model = {
            "num_leaves": meta_best_params["num_leaves"],
            "max_depth": meta_best_params["max_depth"],
            "learning_rate": meta_best_params["learning_rate"],
            "n_estimators": meta_best_params["n_estimators"],
            "class_weight": None if meta_best_params["class_weight_choice"]=="none" else {0:1,1:2},
            "random_state": random_state,
            "n_jobs": -1,
        }
        meta_model = lgb.LGBMClassifier(**meta_params_for_model)
        meta_model.fit(meta_X_train, meta_y_train)

        # threshold tuning via nested KFold on meta_X_train
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        meta_val_probs = []
        meta_val_trues = []
        for tr_idx, val_idx in kf.split(meta_X_train):
            mdl = lgb.LGBMClassifier(**meta_params_for_model)
            mdl.fit(meta_X_train[tr_idx], meta_y_train.iloc[tr_idx])
            probs = mdl.predict_proba(meta_X_train[val_idx])[:, 1]
            meta_val_probs.append(probs)
            meta_val_trues.append(meta_y_train.iloc[val_idx].values)
        meta_val_probs = np.concatenate(meta_val_probs)
        meta_val_trues = np.concatenate(meta_val_trues)
        # search threshold
        best_thr, best_thr_f1 = 0.5, -1.0
        for thr in np.linspace(0.0, 1.0, 101):
            preds = (meta_val_probs > thr).astype(int)
            f = f1_score(meta_val_trues, preds, zero_division=0)
            if f > best_thr_f1:
                best_thr_f1 = f
                best_thr = thr
        print("Meta best threshold:", best_thr, "F1:", best_thr_f1)

        # final meta test evaluation
        meta_test_proba = meta_model.predict_proba(meta_X_test)[:, 1]
        meta_test_pred = (meta_test_proba > best_thr).astype(int)
        meta_report = classification_report(y_test, meta_test_pred, target_names=["普通 (0)", "興奮 (1)"], zero_division=0, output_dict=True)

        # AIC for meta
        AIC_meta = compute_aic_from_probs(y_test, meta_test_proba, k=meta_X_train.shape[1])

        # SHAP for meta
        shap_info = None
        if _HAS_SHAP and save_shap:
            try:
                explainer = shap.TreeExplainer(meta_model)
                shap_vals = explainer.shap_values(meta_X_test)
                savepath = f"shap_meta_win{window}.png"
                plt.figure(figsize=(6,4))
                shap.summary_plot(shap_vals, pd.DataFrame(meta_X_test, columns=["rf","lgb","cat"]), show=False)
                plt.tight_layout()
                plt.savefig(savepath, dpi=150)
                plt.close()
                shap_info = f"Saved: {savepath}"
            except Exception as e:
                shap_info = f"SHAP failed: {e}"
        else:
            shap_info = "shap not installed or disabled"

        # record
        results.append({
            "window": window,
            "rf_best_cv_f1": float(rf_study.best_value) if 'rf_study' in locals() else np.nan,
            "lgb_best_cv_f1": float(lgb_study.best_value) if 'lgb_study' in locals() else np.nan,
            "cat_best_cv_f1": float(cat_study.best_value) if 'cat_study' in locals() else np.nan,
            "meta_best_cv_f1": float(meta_study.best_value),
            "rf_test_f1": rf_report["興奮 (1)"]["f1-score"] if rf_report is not None else np.nan,
            "lgb_test_f1": lgb_report["興奮 (1)"]["f1-score"] if lgb is not None and lgb_report is not None else np.nan,
            "cat_test_f1": cat_report["興奮 (1)"]["f1-score"] if CatBoostClassifier is not None and cat_report is not None else np.nan,
            "meta_test_f1": meta_report["興奮 (1)"]["f1-score"],
            "meta_test_accuracy": meta_report["accuracy"],
            "meta_test_recall_exciting": meta_report["興奮 (1)"]["recall"],
            "meta_test_precision_exciting": meta_report["興奮 (1)"]["precision"],
            "AIC_meta": float(AIC_meta),
            "meta_best_threshold": float(best_thr),
            "pca_used": bool(pca_used),
            "shap_info": shap_info,
            "meta_best_params": meta_params_for_model,
            "rf_best_params": rf_params_for_model,
            "lgb_best_params": lgb_params_for_model if lgb is not None else None,
            "cat_best_params": cat_params_for_model if CatBoostClassifier is not None else None,
        })

        # quick print
        print("Window", window, "meta F1:", results[-1]["meta_test_f1"])

    df_res = pd.DataFrame(results)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n=== Final Summary ===")
    print(df_res.to_string(index=False))
    return df_res

# -------------
# 実行部分
# -------------
if __name__ == "__main__":
    # 必要ならここで探索回数変更
    df_results = run_pipeline(windows=[1,2,3], TOP_N_FEATURES=50, apply_pca_threshold=200, n_trials_base=40, n_trials_meta=40, random_state=42, save_shap=True)
