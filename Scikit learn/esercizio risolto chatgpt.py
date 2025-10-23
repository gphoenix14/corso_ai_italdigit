#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runner degli esperimenti su XGBoost:
- A: effetto sbilanciamento e scale_pos_weight
- B: learning_rate vs n_estimators (OFAT)
- C: complessità albero (max_depth, min_child_weight)
- D: subsample, colsample_bytree e stabilità su semi diversi
- E: rumore/NaN + confronto imputazione median vs mean
- F: RandomizedSearchCV compatta vs estesa (tempo vs qualità)

Output: CSV in ./results e riepilogo a console.
"""

import os
import time
import warnings
from pathlib import Path
from itertools import product
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------
# Import librerie ML
# ---------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("Manca 'xgboost'. Installa con: pip install xgboost") from e

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False


# ---------------------------------------------------------------------
# Utility di base
# ---------------------------------------------------------------------
RANDOM_STATE = 42
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def timed(fn):
    """Decorator per misurare i tempi di esecuzione."""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        return out, time.time() - t0
    return wrapper


def make_dataset(
    n_samples=5000,
    n_num=12,
    informative=6,
    redundant=2,
    weights=(0.85, 0.15),
    class_sep=1.2,
    flip_y=0.01,
    nan_num_pct=0.01,
    nan_cat_pct=0.01,
    seed=RANDOM_STATE,
):
    """Crea dataset misto (numerico+categorico) con sbilanciamento e NaN controllati."""
    rng = np.random.RandomState(seed)
    X_num, y = make_classification(
        n_samples=n_samples,
        n_features=n_num,
        n_informative=informative,
        n_redundant=redundant,
        n_repeated=0,
        n_classes=2,
        weights=list(weights),
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=seed,
    )
    num_cols = [f"num_{i}" for i in range(X_num.shape[1])]
    df_num = pd.DataFrame(X_num, columns=num_cols)

    n = df_num.shape[0]
    # categoriche sintetiche
    cat_sector = rng.choice(["finance", "health", "it", "retail"], size=n, p=[0.25, 0.25, 0.35, 0.15])
    cat_region = rng.choice(["north", "center", "south"], size=n, p=[0.4, 0.2, 0.4])

    df = df_num.copy()
    df["sector"] = cat_sector
    df["region"] = cat_region
    df["target"] = y

    # Introduzione NaN
    for c in num_cols:
        mask = rng.rand(n) < nan_num_pct
        df.loc[mask, c] = np.nan

    for c in ["sector", "region"]:
        mask = rng.rand(n) < nan_cat_pct
        df.loc[mask, c] = np.nan

    return df, num_cols, ["sector", "region"]


def build_preprocess(numeric_features, categorical_features, num_strategy="median"):
    """Costruisce ColumnTransformer per numeriche/categoriche."""
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy=num_strategy))])

    # compat per versioni sklearn: OneHotEncoder(sparse_output) -> fallback a sparse
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        n_jobs=None,
    )
    return preprocess


def xgb_base(scale_pos_weight, seed=RANDOM_STATE):
    """Istanzia un XGBClassifier di base; gli iperparametri verranno poi fissati o cercati."""
    return XGBClassifier(
        booster="gbtree",
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )


def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    """Fit + metriche su test set."""
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "tn": confusion_matrix(y_test, y_pred)[0, 0],
        "fp": confusion_matrix(y_test, y_pred)[0, 1],
        "fn": confusion_matrix(y_test, y_pred)[1, 0],
        "tp": confusion_matrix(y_test, y_pred)[1, 1],
    }
    return metrics


def compute_spw(y_train):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    return neg / max(pos, 1), pos, neg


# ---------------------------------------------------------------------
# Esperimenti
# ---------------------------------------------------------------------
@timed
def experiment_A(weights_list=((0.90, 0.10), (0.85, 0.15), (0.80, 0.20)), seed=RANDOM_STATE):
    """Sbilanciamento e scale_pos_weight con RandomizedSearchCV standard."""
    rows = []
    for weights in weights_list:
        df, num_cols, cat_cols = make_dataset(weights=weights, seed=seed)
        X = df.drop(columns=["target"])
        y = df["target"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

        spw, pos, neg = compute_spw(y_tr)
        preprocess = build_preprocess(num_cols, cat_cols)
        xgb = xgb_base(spw, seed=seed)
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", xgb)])

        # spazio ricerca (come baseline)
        param_distributions = {
            "model__n_estimators": np.arange(150, 901, 50),
            "model__max_depth": np.arange(3, 11),
            "model__learning_rate": np.round(np.logspace(-2, -0.5, 10), 5),
            "model__subsample": np.round(np.linspace(0.5, 1.0, 11), 2),
            "model__colsample_bytree": np.round(np.linspace(0.5, 1.0, 11), 2),
            "model__min_child_weight": [1, 2, 3, 4, 5, 6, 7],
            "model__gamma": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
            "model__reg_lambda": np.round(np.logspace(-2, 2, 10), 5),
            "model__reg_alpha": np.round(np.logspace(-3, 1, 10), 5),
            "model__scale_pos_weight": [spw],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rs = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions,
            n_iter=80,
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv,
            verbose=0,
            refit=True,
            random_state=seed,
        )
        if TQDM_AVAILABLE:
            for _ in tqdm(range(1), desc=f"Exp A weights={weights}", unit="run", leave=False):
                rs.fit(X_tr, y_tr)
        else:
            rs.fit(X_tr, y_tr)

        best = rs.best_estimator_
        metrics = evaluate_pipeline(best, X_tr, y_tr, X_te, y_te)  # refit già eseguito
        rows.append({
            "weights": weights,
            "pos_train": pos,
            "neg_train": neg,
            "scale_pos_weight": spw,
            "cv_best_roc_auc": rs.best_score_,
            **metrics,
            "best_params": rs.best_params_,
        })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(RESULTS_DIR / "experiment_A_imbalance.csv", index=False)
    return df_res


@timed
def experiment_B(seed=RANDOM_STATE):
    """OFAT: learning_rate vs n_estimators, senza random search."""
    df, num_cols, cat_cols = make_dataset(seed=seed)
    X = df.drop(columns=["target"])
    y = df["target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    spw, _, _ = compute_spw(y_tr)
    preprocess = build_preprocess(num_cols, cat_cols)

    learning_rates = [0.01, 0.05, 0.1, 0.2]
    n_estimators_list = [200, 400, 800]

    rows = []
    it = product(learning_rates, n_estimators_list)
    it = list(it)
    iterator = tqdm(it, desc="Exp B OFAT", unit="cfg", leave=False) if TQDM_AVAILABLE else it

    for lr, ne in iterator:
        xgb = xgb_base(spw, seed=seed)
        xgb.set_params(learning_rate=lr, n_estimators=ne, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                       min_child_weight=2, gamma=0.1, reg_lambda=1.0, reg_alpha=0.1)
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", xgb)])
        metrics = evaluate_pipeline(pipe, X_tr, y_tr, X_te, y_te)
        rows.append({"learning_rate": lr, "n_estimators": ne, **metrics})

    df_res = pd.DataFrame(rows)
    df_res.to_csv(RESULTS_DIR / "experiment_B_lr_n_estimators.csv", index=False)
    return df_res


@timed
def experiment_C(seed=RANDOM_STATE):
    """OFAT: max_depth x min_child_weight."""
    df, num_cols, cat_cols = make_dataset(seed=seed)
    X = df.drop(columns=["target"])
    y = df["target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    spw, _, _ = compute_spw(y_tr)
    preprocess = build_preprocess(num_cols, cat_cols)

    max_depths = [3, 5, 7, 9]
    min_child_weights = [1, 3, 5]

    rows = []
    it = product(max_depths, min_child_weights)
    it = list(it)
    iterator = tqdm(it, desc="Exp C tree complexity", unit="cfg", leave=False) if TQDM_AVAILABLE else it

    for md, mcw in iterator:
        xgb = xgb_base(spw, seed=seed)
        xgb.set_params(learning_rate=0.05, n_estimators=400, max_depth=md, min_child_weight=mcw,
                       subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_lambda=1.0, reg_alpha=0.1)
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", xgb)])
        metrics = evaluate_pipeline(pipe, X_tr, y_tr, X_te, y_te)
        rows.append({"max_depth": md, "min_child_weight": mcw, **metrics})

    df_res = pd.DataFrame(rows)
    df_res.to_csv(RESULTS_DIR / "experiment_C_tree_complexity.csv", index=False)
    return df_res


@timed
def experiment_D(seeds=(13, 42, 123)):
    """Stabilità: subsample x colsample_bytree su 3 semi diversi."""
    df, num_cols, cat_cols = make_dataset(seed=RANDOM_STATE)
    X = df.drop(columns=["target"])
    y = df["target"].values

    subs = [0.6, 0.8, 1.0]
    cols = [0.6, 0.8, 1.0]

    rows = []
    iterator = product(subs, cols, seeds)
    iterator = list(iterator)
    iterator = tqdm(iterator, desc="Exp D stability", unit="run", leave=False) if TQDM_AVAILABLE else iterator

    for sub, col, seed in iterator:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        spw, _, _ = compute_spw(y_tr)
        preprocess = build_preprocess(num_cols, cat_cols)
        xgb = xgb_base(spw, seed=seed)
        xgb.set_params(learning_rate=0.05, n_estimators=400, max_depth=5, min_child_weight=2,
                       subsample=sub, colsample_bytree=col, gamma=0.1, reg_lambda=1.0, reg_alpha=0.1)
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", xgb)])
        metrics = evaluate_pipeline(pipe, X_tr, y_tr, X_te, y_te)
        rows.append({"subsample": sub, "colsample_bytree": col, "seed": seed, **metrics})

    df_res = pd.DataFrame(rows)
    # statistiche di stabilità (std su semi)
    agg = df_res.groupby(["subsample", "colsample_bytree"]).agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    ).reset_index()
    df_res.to_csv(RESULTS_DIR / "experiment_D_stability_raw.csv", index=False)
    agg.to_csv(RESULTS_DIR / "experiment_D_stability_agg.csv", index=False)
    return agg


@timed
def experiment_E(seed=RANDOM_STATE):
    """Rumore/NaN: confronto imputazione numerica median vs mean con NaN aumentati."""
    configs = [
        {"nan_num_pct": 0.05, "nan_cat_pct": 0.03, "num_strategy": "median"},
        {"nan_num_pct": 0.05, "nan_cat_pct": 0.03, "num_strategy": "mean"},
    ]

    rows = []
    iterator = tqdm(configs, desc="Exp E NaN", unit="cfg", leave=False) if TQDM_AVAILABLE else configs

    for cfg in iterator:
        df, num_cols, cat_cols = make_dataset(
            seed=seed, nan_num_pct=cfg["nan_num_pct"], nan_cat_pct=cfg["nan_cat_pct"]
        )
        X = df.drop(columns=["target"])
        y = df["target"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        spw, _, _ = compute_spw(y_tr)
        preprocess = build_preprocess(num_cols, cat_cols, num_strategy=cfg["num_strategy"])
        xgb = xgb_base(spw, seed=seed)
        xgb.set_params(learning_rate=0.05, n_estimators=400, max_depth=5, min_child_weight=2,
                       subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_lambda=1.0, reg_alpha=0.1)
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", xgb)])
        metrics = evaluate_pipeline(pipe, X_tr, y_tr, X_te, y_te)
        rows.append({**cfg, **metrics})

    df_res = pd.DataFrame(rows)
    df_res.to_csv(RESULTS_DIR / "experiment_E_nan_imputation.csv", index=False)
    return df_res


@timed
def experiment_F(seed=RANDOM_STATE):
    """RandomizedSearchCV compatta vs estesa: qualità vs tempo."""
    df, num_cols, cat_cols = make_dataset(seed=seed)
    X = df.drop(columns=["target"])
    y = df["target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    spw, _, _ = compute_spw(y_tr)
    preprocess = build_preprocess(num_cols, cat_cols)
    base = xgb_base(spw, seed=seed)

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", base)])

    search_spaces = {
        "compact": {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__gamma": [0.0, 0.1, 0.3],
            "model__reg_lambda": [0.1, 1.0, 10.0],
            "model__reg_alpha": [0.0, 0.1, 1.0],
            "model__scale_pos_weight": [spw],
        },
        "extended": {
            "model__n_estimators": np.arange(150, 901, 50),
            "model__max_depth": np.arange(3, 11),
            "model__learning_rate": np.round(np.logspace(-2, -0.5, 10), 5),
            "model__subsample": np.round(np.linspace(0.5, 1.0, 11), 2),
            "model__colsample_bytree": np.round(np.linspace(0.5, 1.0, 11), 2),
            "model__min_child_weight": [1, 2, 3, 4, 5, 6, 7],
            "model__gamma": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
            "model__reg_lambda": np.round(np.logspace(-2, 2, 10), 5),
            "model__reg_alpha": np.round(np.logspace(-3, 1, 10), 5),
            "model__scale_pos_weight": [spw],
        },
    }
    n_iters = {"compact": 30, "extended": 120}

    rows = []
    for tag in ["compact", "extended"]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rs = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=search_spaces[tag],
            n_iter=n_iters[tag],
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv,
            verbose=0,
            refit=True,
            random_state=seed,
        )
        if TQDM_AVAILABLE:
            for _ in tqdm(range(1), desc=f"Exp F {tag}", unit="run", leave=False):
                rs.fit(X_tr, y_tr)
        else:
            rs.fit(X_tr, y_tr)

        best = rs.best_estimator_
        metrics = evaluate_pipeline(best, X_tr, y_tr, X_te, y_te)
        rows.append({
            "mode": tag,
            "n_iter": n_iters[tag],
            "cv_best_roc_auc": rs.best_score_,
            **metrics,
            "best_params": rs.best_params_
        })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(RESULTS_DIR / "experiment_F_search_depth.csv", index=False)
    return df_res


# ---------------------------------------------------------------------
# Riepilogo e lancio
# ---------------------------------------------------------------------
def print_quick_summary(df: pd.DataFrame, label: str, extra_cols=()):
    """Stampa best performer per ROC-AUC e F1."""
    print(f"\n=== {label}: Best performer ===")
    if "roc_auc" in df.columns:
        idx = df["roc_auc"].idxmax()
        row = df.loc[idx]
        print(f"- Miglior ROC-AUC: {row['roc_auc']:.4f} | "
              f"Acc {row.get('accuracy', np.nan):.4f} | F1 {row.get('f1', np.nan):.4f}")
        if extra_cols:
            print("  Extra:", {c: row[c] for c in extra_cols if c in df.columns})
    if "f1" in df.columns:
        idx = df["f1"].idxmax()
        row = df.loc[idx]
        print(f"- Miglior F1     : {row['f1']:.4f} | "
              f"Acc {row.get('accuracy', np.nan):.4f} | ROC-AUC {row.get('roc_auc', np.nan):.4f}")
        if extra_cols:
            print("  Extra:", {c: row[c] for c in extra_cols if c in df.columns})


def main():
    print(">> Avvio esperimenti XGBoost. Output CSV in ./results\n")

    (dfA, tA) = experiment_A()
    dfA.to_csv(RESULTS_DIR / "experiment_A_imbalance.csv", index=False)
    print(f"Esperimento A completato in {tA:.1f}s; file: experiment_A_imbalance.csv")
    print_quick_summary(dfA, "Esperimento A", extra_cols=("weights", "scale_pos_weight", "cv_best_roc_auc"))

    (dfB, tB) = experiment_B()
    dfB.to_csv(RESULTS_DIR / "experiment_B_lr_n_estimators.csv", index=False)
    print(f"\nEsperimento B completato in {tB:.1f}s; file: experiment_B_lr_n_estimators.csv")
    print_quick_summary(dfB, "Esperimento B", extra_cols=("learning_rate", "n_estimators"))

    (dfC, tC) = experiment_C()
    dfC.to_csv(RESULTS_DIR / "experiment_C_tree_complexity.csv", index=False)
    print(f"\nEsperimento C completato in {tC:.1f}s; file: experiment_C_tree_complexity.csv")
    print_quick_summary(dfC, "Esperimento C", extra_cols=("max_depth", "min_child_weight"))

    (dfD, tD) = experiment_D()
    dfD.to_csv(RESULTS_DIR / "experiment_D_stability_agg.csv", index=False)
    print(f"\nEsperimento D completato in {tD:.1f}s; file: experiment_D_stability_agg.csv")
    print_quick_summary(dfD.rename(columns={"roc_auc_mean":"roc_auc","f1_mean":"f1"}), "Esperimento D (media)", extra_cols=("subsample", "colsample_bytree"))

    (dfE, tE) = experiment_E()
    dfE.to_csv(RESULTS_DIR / "experiment_E_nan_imputation.csv", index=False)
    print(f"\nEsperimento E completato in {tE:.1f}s; file: experiment_E_nan_imputation.csv")
    print_quick_summary(dfE, "Esperimento E", extra_cols=("num_strategy", "nan_num_pct", "nan_cat_pct"))

    (dfF, tF) = experiment_F()
    dfF.to_csv(RESULTS_DIR / "experiment_F_search_depth.csv", index=False)
    print(f"\nEsperimento F completato in {tF:.1f}s; file: experiment_F_search_depth.csv")
    print_quick_summary(dfF, "Esperimento F", extra_cols=("mode", "n_iter", "cv_best_roc_auc"))

    print("\n>> Completato. Controlla la cartella ./results per i CSV dettagliati.")


if __name__ == "__main__":
    main()
