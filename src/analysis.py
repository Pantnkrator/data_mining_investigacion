"""
Detección de Anomalías en Logs de un Sistema de Reportes de Ventas
Mediante Algoritmos de Minería de Datos: Isolation Forest, LOF y One-Class SVM
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(ROOT_DIR, "sales_report_system_logs.csv")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
LATEX_FIGURES = os.path.join(ROOT_DIR, "latex", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LATEX_FIGURES, exist_ok=True)

PALETTE_NORMAL = "#4C72B0"
PALETTE_ANOMALY = "#DD8452"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────
# 1. CARGA Y EXPLORACIÓN INICIAL
# ─────────────────────────────────────────────

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday
    return df


def print_eda_summary(df):
    print("=" * 60)
    print("RESUMEN EXPLORATORIO DE DATOS")
    print("=" * 60)
    print(f"Total de registros       : {len(df):,}")
    print(f"Período                  : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"Servicios únicos         : {df['service'].nunique()}")
    print(f"Trazas únicas            : {df['trace_id'].nunique():,}")
    print(f"Usuarios únicos          : {df['user_hash'].nunique()}")
    print(f"Tenants únicos           : {df['tenant_hash'].nunique()}")
    print()
    print("Distribución por servicio:")
    print(df["service"].value_counts().to_string())
    print()
    print("Distribución de anomalías:")
    counts = df["is_anomaly"].value_counts()
    print(f"  Normales  : {counts[0]:,} ({counts[0]/len(df)*100:.2f}%)")
    print(f"  Anomalías : {counts[1]:,} ({counts[1]/len(df)*100:.2f}%)")
    print()
    print("Tipos de anomalía:")
    print(df["anomaly_type"].value_counts().to_string())
    print()
    print("Estadísticas numéricas (report-service):")
    rs = df[df["service"] == "report-service"]
    print(rs[["response_time_ms", "db_query_time_ms", "records_returned",
              "cpu_usage_percent", "memory_usage_mb"]].describe().round(2).to_string())


# ─────────────────────────────────────────────
# 2. FIGURAS EDA
# ─────────────────────────────────────────────

def fig_anomaly_distribution(df):
    """Distribución de tipos de anomalía."""
    counts = df[df["is_anomaly"] == 1].groupby(["service", "anomaly_type"]).size().reset_index(name="count")
    pivot = counts.pivot(index="anomaly_type", columns="service", values="count").fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel izquierdo: total por tipo
    type_counts = df["anomaly_type"].value_counts()
    colors = [PALETTE_ANOMALY if t != "normal" else PALETTE_NORMAL for t in type_counts.index]
    axes[0].barh(type_counts.index, type_counts.values, color=colors)
    axes[0].set_xlabel("Cantidad de registros")
    axes[0].set_title("Distribución general por tipo de anomalía")
    for i, v in enumerate(type_counts.values):
        axes[0].text(v + 50, i, f"{v:,}", va="center", fontsize=9)

    # Panel derecho: por servicio
    pivot.plot(kind="bar", ax=axes[1], colormap="tab10", width=0.7)
    axes[1].set_xlabel("Tipo de anomalía")
    axes[1].set_ylabel("Cantidad")
    axes[1].set_title("Anomalías por tipo y servicio")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="Servicio", fontsize=8)

    plt.tight_layout()
    _save(fig, "fig_anomaly_distribution")


def fig_response_time_boxplot(df):
    """Boxplot de tiempos de respuesta por tipo de anomalía."""
    rs = df[df["service"] == "report-service"].copy()
    order = ["normal", "slow_database_query", "high_error_rate",
             "high_resource_usage", "large_report_export"]
    labels = {
        "normal": "Normal",
        "slow_database_query": "DB lenta",
        "high_error_rate": "Alta tasa\nde errores",
        "high_resource_usage": "Alto uso\nde recursos",
        "large_report_export": "Exportación\ngrande"
    }
    rs["anomaly_label"] = rs["anomaly_type"].map(labels)
    order_labels = [labels[o] for o in order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    palette = {labels["normal"]: PALETTE_NORMAL}
    for k in order[1:]:
        palette[labels[k]] = PALETTE_ANOMALY

    sns.boxplot(data=rs, x="anomaly_label", y="response_time_ms",
                order=order_labels, palette=palette, ax=axes[0],
                showfliers=False, width=0.5)
    axes[0].set_xlabel("Tipo de anomalía")
    axes[0].set_ylabel("Tiempo de respuesta (ms)")
    axes[0].set_title("Tiempo de respuesta por tipo (report-service)")

    sns.boxplot(data=rs, x="anomaly_label", y="db_query_time_ms",
                order=order_labels, palette=palette, ax=axes[1],
                showfliers=False, width=0.5)
    axes[1].set_xlabel("Tipo de anomalía")
    axes[1].set_ylabel("Tiempo de consulta DB (ms)")
    axes[1].set_title("Tiempo de consulta DB por tipo (report-service)")

    plt.tight_layout()
    _save(fig, "fig_response_time_boxplot")


def fig_correlation_heatmap(df):
    """Matriz de correlación de features numéricas."""
    rs = df[df["service"] == "report-service"]
    feats = ["response_time_ms", "db_query_time_ms", "api_response_time_ms",
             "records_returned", "rows_scanned", "report_size_kb",
             "cpu_usage_percent", "memory_usage_mb", "retry_count",
             "error_count", "is_anomaly"]
    corr = rs[feats].corr()

    feat_labels = {
        "response_time_ms": "T. respuesta",
        "db_query_time_ms": "T. consulta DB",
        "api_response_time_ms": "T. respuesta API",
        "records_returned": "Registros retornados",
        "rows_scanned": "Filas escaneadas",
        "report_size_kb": "Tamaño reporte (KB)",
        "cpu_usage_percent": "CPU (%)",
        "memory_usage_mb": "Memoria (MB)",
        "retry_count": "Reintentos",
        "error_count": "Errores",
        "is_anomaly": "Es anomalía"
    }
    corr.index = [feat_labels[c] for c in corr.index]
    corr.columns = [feat_labels[c] for c in corr.columns]

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7},
                linewidths=0.3)
    ax.set_title("Correlación entre variables numéricas (report-service)")
    plt.tight_layout()
    _save(fig, "fig_correlation_heatmap")


def fig_temporal_patterns(df):
    """Patrones temporales: anomalías por hora del día."""
    hourly = df.groupby(["hour", "is_anomaly"]).size().reset_index(name="count")
    normal_h = hourly[hourly["is_anomaly"] == 0].set_index("hour")["count"]
    anomaly_h = hourly[hourly["is_anomaly"] == 1].set_index("hour")["count"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].fill_between(normal_h.index, normal_h.values, alpha=0.6,
                         color=PALETTE_NORMAL, label="Normal")
    axes[0].fill_between(anomaly_h.index, anomaly_h.values, alpha=0.7,
                         color=PALETTE_ANOMALY, label="Anomalía")
    axes[0].set_xlabel("Hora del día")
    axes[0].set_ylabel("Cantidad de registros")
    axes[0].set_title("Distribución temporal de registros")
    axes[0].legend()
    axes[0].set_xticks(range(0, 24, 2))

    # Tasa de anomalía por hora
    rate = anomaly_h / (normal_h + anomaly_h) * 100
    axes[1].bar(rate.index, rate.values, color=PALETTE_ANOMALY, alpha=0.8)
    axes[1].axhline(rate.mean(), color="black", linestyle="--", linewidth=1,
                    label=f"Media: {rate.mean():.1f}%")
    axes[1].set_xlabel("Hora del día")
    axes[1].set_ylabel("Tasa de anomalías (%)")
    axes[1].set_title("Tasa de anomalías por hora")
    axes[1].legend()
    axes[1].set_xticks(range(0, 24, 2))

    plt.tight_layout()
    _save(fig, "fig_temporal_patterns")


# ─────────────────────────────────────────────
# 3. PREPROCESAMIENTO
# ─────────────────────────────────────────────

NUMERIC_FEATURES = [
    "response_time_ms", "db_query_time_ms", "api_response_time_ms",
    "records_returned", "rows_scanned", "report_size_kb",
    "cpu_usage_percent", "memory_usage_mb", "retry_count", "error_count"
]

SERVICE_ORDER = ["report-service", "sales-api", "database"]


def preprocess(df):
    le = LabelEncoder()
    df = df.copy()
    df["service_enc"] = le.fit_transform(df["service"])
    df["report_type_enc"] = LabelEncoder().fit_transform(df["report_type"])
    df["user_role_enc"] = LabelEncoder().fit_transform(df["user_role"])

    features = NUMERIC_FEATURES + ["service_enc", "report_type_enc", "user_role_enc", "hour"]
    X = df[features].copy()
    X = X.fillna(0)
    y = df["is_anomaly"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, features, scaler


# ─────────────────────────────────────────────
# 4. MODELOS
# ─────────────────────────────────────────────

def train_models(X_train_normal):
    """Entrena los 3 modelos solo con datos normales."""
    models = {
        "Isolation Forest": IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=30,
            contamination=0.05,
            novelty=True,
            n_jobs=-1
        ),
        "One-Class SVM": OneClassSVM(
            kernel="rbf",
            nu=0.05,
            gamma="scale"
        )
    }

    print("\nEntrenando modelos...")
    for name, model in models.items():
        model.fit(X_train_normal)
        print(f"  [{name}] entrenado.")

    return models


def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        # sklearn: 1=normal, -1=anomaly; convert to our labels
        raw_pred = model.predict(X_test)
        y_pred = np.where(raw_pred == -1, 1, 0)

        # score: negative = more anomalous
        if hasattr(model, "score_samples"):
            scores = -model.score_samples(X_test)
        elif hasattr(model, "decision_function"):
            scores = -model.decision_function(X_test)
        else:
            scores = y_pred.astype(float)

        # Normalize scores for ROC
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, pos_label=1, average="binary", zero_division=0
        )
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "y_pred": y_pred,
            "scores": scores,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "cm": cm
        }

        print(f"\n  [{name}]")
        print(f"    Precision : {precision:.4f}")
        print(f"    Recall    : {recall:.4f}")
        print(f"    F1-Score  : {f1:.4f}")
        print(f"    AUC-ROC   : {roc_auc:.4f}")

    return results


# ─────────────────────────────────────────────
# 5. FIGURAS DE RESULTADOS
# ─────────────────────────────────────────────

def fig_roc_curves(results):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for (name, res), color in zip(results.items(), colors):
        ax.plot(res["fpr"], res["tpr"], color=color, lw=2,
                label=f"{name} (AUC = {res['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Clasificador aleatorio")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.01])
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curvas ROC — Comparación de Modelos")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_roc_curves")


def fig_confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["cm"]
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
                          for row_v, row_p in zip(cm, cm_pct)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Anomalía"],
                    yticklabels=["Normal", "Anomalía"],
                    cbar=False, linewidths=0.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
    plt.suptitle("Matrices de Confusión", fontsize=11, y=1.01)
    plt.tight_layout()
    _save(fig, "fig_confusion_matrices")


def fig_metrics_comparison(results):
    names = list(results.keys())
    metrics = ["precision", "recall", "f1", "auc"]
    labels = ["Precisión", "Recall", "F1-Score", "AUC-ROC"]
    colors = ["#5C85D6", "#E07B54", "#5DB85D", "#B45DB8"]

    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (m, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + i * width - 1.5 * width, vals, width,
                      label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Valor")
    ax.set_title("Comparación de métricas por modelo")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_metrics_comparison")


def fig_feature_importance(df, scaler, features):
    """Importancia de variables para Isolation Forest (mean depth)."""
    X_all, y_all, _, _ = preprocess(df)

    # Train on all data
    iforest = IsolationForest(n_estimators=200, contamination=0.05,
                               random_state=42, n_jobs=-1)
    iforest.fit(X_all)

    # Permutation-style importance: measure score drop per feature
    base_scores = iforest.score_samples(X_all)
    importances = []
    rng = np.random.RandomState(0)
    for i in range(X_all.shape[1]):
        X_perm = X_all.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        perm_scores = iforest.score_samples(X_perm)
        importances.append(np.mean(base_scores - perm_scores))

    feat_labels_short = {
        "response_time_ms": "T. respuesta",
        "db_query_time_ms": "T. consulta DB",
        "api_response_time_ms": "T. respuesta API",
        "records_returned": "Registros",
        "rows_scanned": "Filas escaneadas",
        "report_size_kb": "Tamaño reporte",
        "cpu_usage_percent": "CPU %",
        "memory_usage_mb": "Memoria",
        "retry_count": "Reintentos",
        "error_count": "Errores",
        "service_enc": "Servicio",
        "report_type_enc": "Tipo reporte",
        "user_role_enc": "Rol usuario",
        "hour": "Hora"
    }

    labels = [feat_labels_short.get(f, f) for f in features]
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE_ANOMALY if importances[i] > 0 else PALETTE_NORMAL for i in idx]
    ax.barh([labels[i] for i in idx], [importances[i] for i in idx], color=colors)
    ax.set_xlabel("Importancia (caída de puntuación media)")
    ax.set_title("Importancia de variables — Isolation Forest")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    _save(fig, "fig_feature_importance")

    return [(features[i], importances[i]) for i in idx]


def fig_anomaly_per_service(df, results):
    """Detección de anomalías por servicio para el mejor modelo."""
    best_name = max(results, key=lambda n: results[n]["f1"])
    print(f"\nMejor modelo por F1: {best_name}")

    X_all, y_all, features, scaler = preprocess(df)
    model = results[best_name]["_model"]
    raw_pred = model.predict(X_all)
    y_pred_all = np.where(raw_pred == -1, 1, 0)
    df = df.copy()
    df["predicted_anomaly"] = y_pred_all

    per_service = df.groupby("service").apply(
        lambda g: pd.Series({
            "total": len(g),
            "real_anomaly": g["is_anomaly"].sum(),
            "pred_anomaly": g["predicted_anomaly"].sum(),
            "true_positive": ((g["is_anomaly"] == 1) & (g["predicted_anomaly"] == 1)).sum(),
        })
    ).reindex(SERVICE_ORDER)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(per_service))
    w = 0.25
    ax.bar(x - w, per_service["real_anomaly"], w, label="Anomalías reales",
           color=PALETTE_ANOMALY, alpha=0.85)
    ax.bar(x, per_service["pred_anomaly"], w, label="Anomalías predichas",
           color="#8C4CA8", alpha=0.85)
    ax.bar(x + w, per_service["true_positive"], w, label="Verdaderos positivos",
           color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(SERVICE_ORDER)
    ax.set_ylabel("Cantidad de registros")
    ax.set_title(f"Detección por servicio — {best_name}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_anomaly_per_service")

    return per_service


# ─────────────────────────────────────────────
# 6. TABLA RESUMEN
# ─────────────────────────────────────────────

def print_results_table(results):
    print("\n" + "=" * 60)
    print("TABLA RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"{'Modelo':<25} {'Precisión':>10} {'Recall':>10} {'F1-Score':>10} {'AUC-ROC':>10}")
    print("-" * 65)
    for name, res in results.items():
        print(f"{name:<25} {res['precision']:>10.4f} {res['recall']:>10.4f} "
              f"{res['f1']:>10.4f} {res['auc']:>10.4f}")


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _save(fig, name):
    for ext in ["pdf", "png"]:
        path_src = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        path_latex = os.path.join(LATEX_FIGURES, f"{name}.{ext}")
        fig.savefig(path_src, bbox_inches="tight")
        fig.savefig(path_latex, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {name}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("Cargando datos...")
    df = load_data()
    print_eda_summary(df)

    print("\nGenerando figuras EDA...")
    fig_anomaly_distribution(df)
    fig_response_time_boxplot(df)
    fig_correlation_heatmap(df)
    fig_temporal_patterns(df)

    print("\nPreprocesando datos...")
    X, y, features, scaler = preprocess(df)

    # Split: train solo con normales, test con todo
    idx_normal = np.where(y == 0)[0]
    idx_test = np.arange(len(y))

    # 70% normales para entrenar
    np.random.seed(42)
    train_idx = np.random.choice(idx_normal, size=int(len(idx_normal) * 0.7), replace=False)
    X_train_normal = X[train_idx]
    X_test = X
    y_test = y

    print(f"\nMuestras de entrenamiento (solo normales): {len(X_train_normal):,}")
    print(f"Muestras de evaluación (total)           : {len(X_test):,}")

    print("\nEntrenando modelos de detección de anomalías...")
    models = train_models(X_train_normal)

    print("\nEvaluando modelos...")
    results = evaluate_models(models, X_test, y_test)

    # Guardar referencia al modelo para fig_anomaly_per_service
    for name in results:
        results[name]["_model"] = models[name]

    print_results_table(results)

    print("\nGenerando figuras de resultados...")
    fig_roc_curves(results)
    fig_confusion_matrices(results)
    fig_metrics_comparison(results)
    imp = fig_feature_importance(df, scaler, features)
    fig_anomaly_per_service(df, results)

    print("\nTop 5 variables más importantes:")
    for feat, val in imp[:5]:
        print(f"  {feat}: {val:.6f}")

    print("\nAnálisis completado. Figuras guardadas en:")
    print(f"  {FIGURES_DIR}")
    print(f"  {LATEX_FIGURES}")


if __name__ == "__main__":
    main()
