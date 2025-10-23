#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 0) Import dipendenze
# =============================================================================
try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit(
        "Manca 'xgboost'. Installa con: pip install xgboost"
    ) from e

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

# =============================================================================
# 1) Generazione dataset sintetico (numerico + categorico) con sbilanciamento
# =============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Dataset numerico base
X_num, y = make_classification(
    n_samples=5000,
    n_features=12,
    n_informative=6,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    weights=[0.85, 0.15],  # sbilanciato
    class_sep=1.2,
    flip_y=0.01,
    random_state=RANDOM_STATE,
)

num_cols = [f"num_{i}" for i in range(X_num.shape[1])]
df_num = pd.DataFrame(X_num, columns=num_cols)

# Aggiungo feature categoriche sintetiche
n = df_num.shape[0]
cat_sector = np.random.choice(["finance", "health", "it", "retail"], size=n, p=[0.25, 0.25, 0.35, 0.15])
cat_region = np.random.choice(["north", "center", "south"], size=n, p=[0.4, 0.2, 0.4])

df = df_num.copy()
df["sector"] = cat_sector
df["region"] = cat_region
df["target"] = y

# Introduco qualche NaN realistico (1% numeriche, 1% categoriche)
for c in num_cols:
    mask = np.random.rand(n) < 0.01
    df.loc[mask, c] = np.nan

for c in ["sector", "region"]:
    mask = np.random.rand(n) < 0.01
    df.loc[mask, c] = np.nan

# =============================================================================
# 2) Train/Test split stratificato
# =============================================================================
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# =============================================================================
# 3) Pre-processing: imputazione numeriche/categoriche + OneHot
# =============================================================================
numeric_features = num_cols
categorical_features = ["sector", "region"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # Nessuna scalatura: XGBoost su alberi non necessita di scaling
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
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

# =============================================================================
# 4) Stima del peso per classi sbilanciate (scale_pos_weight)
# =============================================================================
# XGBoost usa scale_pos_weight = (#negativi / #positivi) sul TRAIN
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_w = neg / max(pos, 1)

# =============================================================================
# 5) Pipeline modello
# =============================================================================
xgb = XGBClassifier(
    booster="gbtree",
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",          # veloce e robusto
    random_state=RANDOM_STATE,
    n_jobs=-1,
    # metto una base, poi RandomizedSearchCV sovrascrive
    scale_pos_weight=scale_pos_w,
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", xgb),
    ]
)

# =============================================================================
# 6) Spazio di ricerca RandomizedSearchCV (solo liste => no dip. da scipy)
# =============================================================================
param_distributions = {
    "model__n_estimators": np.arange(150, 901, 50),             # 150..900
    "model__max_depth": np.arange(3, 11),                        # 3..10
    "model__learning_rate": np.round(np.logspace(-2, -0.5, 10), 5),  # ~0.01..0.316
    "model__subsample": np.round(np.linspace(0.5, 1.0, 11), 2),  # 0.50..1.00
    "model__colsample_bytree": np.round(np.linspace(0.5, 1.0, 11), 2),
    "model__min_child_weight": [1, 2, 3, 4, 5, 6, 7],
    "model__gamma": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    "model__reg_lambda": np.round(np.logspace(-2, 2, 10), 5),   # 0.01..100
    "model__reg_alpha": np.round(np.logspace(-3, 1, 10), 5),    # 0.001..10
    # fisso scale_pos_weight calcolato dal training per coerenza tra fold
    "model__scale_pos_weight": [scale_pos_w],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=80,                 # aumentare per ricerche più approfondite
    scoring="roc_auc",
    n_jobs=-1,
    cv=cv,
    verbose=1,
    refit=True,
    random_state=RANDOM_STATE,
)

# =============================================================================
# 7) Esecuzione Randomized Search + fit del best estimator
# =============================================================================
random_search.fit(X_train, y_train)

print("\n=== MIGLIORI IPERPARAMETRI TROVATI (CV) ===")
for k, v in random_search.best_params_.items():
    print(f"{k}: {v}")
print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")

best_model = random_search.best_estimator_

# =============================================================================
# 8) Valutazione sul test set
# =============================================================================
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n=== METRICHE SU TEST SET ===")
print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1-score   : {f1:.4f}")
print(f"ROC-AUC    : {roc:.4f}")

print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred, digits=4))
