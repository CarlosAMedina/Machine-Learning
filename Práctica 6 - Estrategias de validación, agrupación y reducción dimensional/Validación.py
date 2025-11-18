# -*- coding: utf-8 -*-
"""Practica 6: Implementación de Estrategias de Validación Personalizadas con KNN
    [Carlos Armando Medina Ascencio]
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_validate, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import defaultdict

# Configuramos KNN y el escalador una sola vez
knn = KNeighborsClassifier(n_neighbors=5)
scaler = StandardScaler()

# Diccionario de Datasets
datasets = {
    "Iris": load_iris(),
    "Wine": load_wine(),
    "Breast Cancer": load_breast_cancer()
}

# Diccionario para almacenar los resultados
resultados_por_dataset = defaultdict(list)

def calcular_metricas(y_true, y_pred):
    """Calcula las 3 métricas de rendimiento para un conjunto de predicciones."""
    # Usamos 'weighted' para multiclase y zero_division=0 para suprimir el error matemático.
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    return {'Exactitud': accuracy, 'Precisión': precision, 'Exhaustividad': recall}

def registrar_resultado(dataset, estrategia, metricas):
    """Añade los resultados de una estrategia al diccionario global."""
    resultados_por_dataset[dataset].append({
        'Estrategia': estrategia,
        'Exactitud': metricas['Exactitud'],
        'Precisión': metricas['Precisión'],
        'Exhaustividad': metricas['Exhaustividad']
    })

def aplicar_estrategias_personalizadas(nombre_dataset, data):
    X = data.data
    y = data.target
    X = scaler.fit_transform(X)

    # --- 1. FOLD (80/20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    registrar_resultado(nombre_dataset, 'Fold (80/20)', calcular_metricas(y_test, y_pred))

    # --- 2. EXPLORATION (100%) - Resubstitution ---
    knn.fit(X, y)
    y_pred_exp = knn.predict(X)
    registrar_resultado(nombre_dataset, 'Exploration (100%)', calcular_metricas(y, y_pred_exp))

    # --- Ignorar advertencias durante la validación cruzada para una salida limpia ---
    warnings.filterwarnings('ignore', category=UserWarning)

    # --- 3. LOU (Leave-One-Out) ---
    loo = LeaveOneOut()
    scores_loo = cross_validate(knn, X, y, cv=loo, scoring=['accuracy', 'precision_weighted', 'recall_weighted'])
    metricas_loo = {'Exactitud': scores_loo['test_accuracy'].mean(),
                    'Precisión': scores_loo['test_precision_weighted'].mean(),
                    'Exhaustividad': scores_loo['test_recall_weighted'].mean()}
    registrar_resultado(nombre_dataset, 'LOU (Leave-One-Out)', metricas_loo)

    # --- 4. CROSS VALIDATION (K-Fold) ---
    k_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kcv = cross_validate(knn, X, y, cv=k_cv, scoring=['accuracy', 'precision_weighted', 'recall_weighted'])
    metricas_kcv = {'Exactitud': scores_kcv['test_accuracy'].mean(),
                    'Precisión': scores_kcv['test_precision_weighted'].mean(),
                    'Exhaustividad': scores_kcv['test_recall_weighted'].mean()}
    registrar_resultado(nombre_dataset, 'Cross Validation (K-Fold)', metricas_kcv)

    # Restaurar advertencias
    warnings.filterwarnings('default', category=UserWarning)

# Ejecutar las validaciones para cada dataset
for name, data in datasets.items():
    aplicar_estrategias_personalizadas(name, data)

print("="*80)
print("             RESUMEN ESTRUCTURADO DE MÉTODOS DE VALIDACIÓN POR DATASET")
print("="*80)

# Iterar sobre los resultados para generar una tabla por dataset
for dataset, resultados in resultados_por_dataset.items():
    print(f"\n### Resultados para el Dataset: {dataset} ###")
    print("-" * (len(dataset) + 30))
    df_resultados = pd.DataFrame(resultados)

    # Aseguramos el orden de las columnas y el formato
    df_resultados = df_resultados[['Estrategia', 'Exactitud', 'Precisión', 'Exhaustividad']]
    print(df_resultados.to_markdown(index=False, floatfmt=".4f"))