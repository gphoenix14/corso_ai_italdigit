#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_knn_with_scalers(k_values=range(1, 21), random_state=42):
    # 1) Caricamento dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names, target_names = wine.feature_names, wine.target_names

    # 2) Train/Test split con stratificazione
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # 3) Definizione degli scaler da confrontare (incluso 'none' baseline)
    scalers = {
        "none": None,
        "standard": StandardScaler(),   # Z-score
        "minmax": MinMaxScaler(feature_range=(0, 1)),
        "robust": RobustScaler()        # mediana e IQR
    }

    # 4) Valutazione per ciascuno scaler e ciascun K
    rows = []
    best_models = {}  # per salvare dettagli del miglior K per scaler

    for scaler_name, scaler in scalers.items():
        best_acc = -np.inf
        best_k = None
        best_y_pred = None

        for k in k_values:
            steps = []
            if scaler is not None:
                steps.append(("scaler", scaler))
            steps.append(("knn", KNeighborsClassifier(n_neighbors=k)))
            pipe = Pipeline(steps)

            # Fit SOLO sul train; lo scaler calcola i parametri sul train
            pipe.fit(X_train, y_train)

            # Predizione sul test trasformato con gli stessi parametri
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            rows.append({
                "scaler": scaler_name,
                "k": k,
                "accuracy": acc
            })

            # Teniamo traccia del migliore per questo scaler
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_y_pred = y_pred

        # Stampa dettagli del migliore per lo scaler corrente
        print(f"\n=== SCALER: {scaler_name.upper()} | Best K = {best_k} | Accuracy = {best_acc:.4f} ===")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, best_y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, best_y_pred, target_names=target_names))

        best_models[scaler_name] = {"k": best_k, "accuracy": best_acc}

    # 5) Tabella risultati (tutti i K)
    results_df = pd.DataFrame(rows).sort_values(by=["scaler", "k"]).reset_index(drop=True)
    print("\n=== Accuracy per scaler e per K ===")
    print(results_df.pivot(index="k", columns="scaler", values="accuracy").round(4))

    # 6) Riepilogo: migliore accuracy per ogni scaler
    summary = pd.DataFrame.from_dict(best_models, orient="index").reset_index()
    summary.columns = ["scaler", "best_k", "best_accuracy"]
    summary = summary.sort_values(by="best_accuracy", ascending=False).reset_index(drop=True)
    print("\n=== Miglior risultato per ciascuno scaler ===")
    print(summary)

    return results_df, summary

if __name__ == "__main__":
    # Esegue la valutazione completa
    _all_results, _summary = evaluate_knn_with_scalers()
