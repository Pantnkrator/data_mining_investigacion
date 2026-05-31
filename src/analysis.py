"""
Detección y Caracterización de Comportamientos Anómalos en Registros
Operativos de una Plataforma de Ventas mediante Isolation Forest
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
CSV_PATH   = os.path.join(ROOT_DIR, "sales_report_system_logs.csv")
FIGURES_SRC   = os.path.join(SCRIPT_DIR, "figures")
FIGURES_LATEX = os.path.join(ROOT_DIR, "latex", "figures")

os.makedirs(FIGURES_SRC, exist_ok=True)
os.makedirs(FIGURES_LATEX, exist_ok=True)

PAL_NORMAL  = "#4C72B0"
PAL_ANOMALY = "#DD8452"
SERVICE_ORDER = ["report-service", "sales-api", "database"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150,
})

NUMERIC_FEATURES = [
    "response_time_ms", "db_query_time_ms", "api_response_time_ms",
    "records_returned", "rows_scanned", "report_size_kb",
    "cpu_usage_percent", "memory_usage_mb", "retry_count", "error_count"
]

# ─────────────────────────────────────────────────────────────────
# 1.  CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"]    = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["day"]     = df["timestamp"].dt.day.fillna(0).astype(int)
    df["weekday"] = df["timestamp"].dt.weekday.fillna(0).astype(int)
    return df


def print_eda_summary(df):
    print("=" * 60)
    print("RESUMEN EXPLORATORIO")
    print("=" * 60)
    print(f"Total de registros : {len(df):,}")
    print(f"Período            : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"Servicios          : {df['service'].unique().tolist()}")
    print(f"Trazas únicas      : {df['trace_id'].nunique():,}")
    print(f"Usuarios           : {df['user_hash'].nunique()}")
    print(f"Tenants            : {df['tenant_hash'].nunique()}")
    print()
    counts = df["is_anomaly"].value_counts()
    total  = len(df)
    print(f"Normales           : {counts[0]:,} ({counts[0]/total*100:.2f}%)")
    print(f"Anomalías          : {counts[1]:,} ({counts[1]/total*100:.2f}%)")
    print()
    print("Tipos de anomalía:")
    print(df["anomaly_type"].value_counts().to_string())
    print()
    rs = df[df["service"] == "report-service"]
    print("Estadísticas descriptivas (report-service):")
    print(rs[NUMERIC_FEATURES].describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────────
# 2.  FIGURAS EDA
# ─────────────────────────────────────────────────────────────────

def fig_anomaly_distribution(df):
    type_counts = df["anomaly_type"].value_counts()
    colors = [PAL_ANOMALY if t != "normal" else PAL_NORMAL for t in type_counts.index]

    # by service (only anomalies)
    counts = (df[df["is_anomaly"] == 1]
              .groupby(["service", "anomaly_type"]).size()
              .reset_index(name="count"))
    pivot = counts.pivot(index="anomaly_type", columns="service", values="count").fillna(0)
    pivot = pivot.reindex(SERVICE_ORDER, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].barh(type_counts.index, type_counts.values, color=colors)
    axes[0].set_xlabel("Cantidad de registros")
    axes[0].set_title("Distribución general por tipo de anomalía")
    for i, v in enumerate(type_counts.values):
        axes[0].text(v + 50, i, f"{v:,}", va="center", fontsize=9)

    pivot.plot(kind="bar", ax=axes[1], colormap="tab10", width=0.7)
    axes[1].set_xlabel("Tipo de anomalía")
    axes[1].set_ylabel("Cantidad")
    axes[1].set_title("Anomalías por tipo y servicio")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="Servicio", fontsize=8)

    plt.tight_layout()
    _save(fig, "fig_anomaly_distribution")


def fig_response_time_boxplot(df):
    rs = df[df["service"] == "report-service"].copy()
    order = ["normal", "slow_database_query", "high_error_rate",
             "high_resource_usage", "large_report_export"]
    labels = {
        "normal":              "Normal",
        "slow_database_query": "DB lenta",
        "high_error_rate":     "Alta tasa\nerrores",
        "high_resource_usage": "Alto uso\nrecursos",
        "large_report_export": "Exportación\ngrande",
    }
    rs["label"] = rs["anomaly_type"].map(labels)
    order_lbl = [labels[o] for o in order]
    pal = {labels["normal"]: PAL_NORMAL}
    for k in order[1:]:
        pal[labels[k]] = PAL_ANOMALY

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=rs, x="label", y="response_time_ms",
                order=order_lbl, palette=pal, ax=axes[0], showfliers=False, width=0.5)
    axes[0].set_xlabel("Tipo de anomalía")
    axes[0].set_ylabel("Tiempo de respuesta (ms)")
    axes[0].set_title("Tiempo de respuesta por tipo (report-service)")

    sns.boxplot(data=rs, x="label", y="db_query_time_ms",
                order=order_lbl, palette=pal, ax=axes[1], showfliers=False, width=0.5)
    axes[1].set_xlabel("Tipo de anomalía")
    axes[1].set_ylabel("Tiempo de consulta BD (ms)")
    axes[1].set_title("Tiempo de consulta BD por tipo (report-service)")

    plt.tight_layout()
    _save(fig, "fig_response_time_boxplot")


def fig_correlation_heatmap(df):
    rs = df[df["service"] == "report-service"]
    feats = NUMERIC_FEATURES + ["is_anomaly"]
    corr  = rs[feats].corr()
    feat_map = {
        "response_time_ms": "T. respuesta", "db_query_time_ms": "T. consulta BD",
        "api_response_time_ms": "T. resp. API", "records_returned": "Registros",
        "rows_scanned": "Filas escaneadas", "report_size_kb": "Tamaño rep.",
        "cpu_usage_percent": "CPU (%)", "memory_usage_mb": "Memoria (MB)",
        "retry_count": "Reintentos", "error_count": "Errores",
        "is_anomaly": "Es anomalía"
    }
    corr.index   = [feat_map[c] for c in corr.index]
    corr.columns = [feat_map[c] for c in corr.columns]

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7},
                linewidths=0.3)
    ax.set_title("Correlación entre variables numéricas (report-service)")
    plt.tight_layout()
    _save(fig, "fig_correlation_heatmap")


def fig_temporal_patterns(df):
    hourly   = df.groupby(["hour", "is_anomaly"]).size().reset_index(name="count")
    normal_h = hourly[hourly["is_anomaly"] == 0].set_index("hour")["count"]
    anom_h   = hourly[hourly["is_anomaly"] == 1].set_index("hour")["count"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].fill_between(normal_h.index, normal_h.values, alpha=0.6, color=PAL_NORMAL, label="Normal")
    axes[0].fill_between(anom_h.index,   anom_h.values,   alpha=0.7, color=PAL_ANOMALY, label="Anomalía")
    axes[0].set_xlabel("Hora del día"); axes[0].set_ylabel("Registros")
    axes[0].set_title("Distribución temporal de registros")
    axes[0].legend(); axes[0].set_xticks(range(0, 24, 2))

    rate = anom_h / (normal_h + anom_h) * 100
    axes[1].bar(rate.index, rate.values, color=PAL_ANOMALY, alpha=0.8)
    axes[1].axhline(rate.mean(), color="black", linestyle="--", linewidth=1,
                    label=f"Media: {rate.mean():.1f}%")
    axes[1].set_xlabel("Hora del día"); axes[1].set_ylabel("Tasa de anomalías (%)")
    axes[1].set_title("Tasa de anomalías por hora del día")
    axes[1].legend(); axes[1].set_xticks(range(0, 24, 2))

    plt.tight_layout()
    _save(fig, "fig_temporal_patterns")

# ─────────────────────────────────────────────────────────────────
# 3.  PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────

def preprocess(df):
    df = df.copy()
    df["service_enc"]     = LabelEncoder().fit_transform(df["service"])
    df["report_type_enc"] = LabelEncoder().fit_transform(df["report_type"])
    df["user_role_enc"]   = LabelEncoder().fit_transform(df["user_role"])

    features = NUMERIC_FEATURES + ["service_enc", "report_type_enc", "user_role_enc", "hour"]
    X = df[features].fillna(0).copy()
    y = df["is_anomaly"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, features, scaler

# ─────────────────────────────────────────────────────────────────
# 4.  PARTICIÓN 70 / 15 / 15
# ─────────────────────────────────────────────────────────────────

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# ─────────────────────────────────────────────────────────────────
# 5.  AJUSTE DEL HIPERPARÁMETRO CONTAMINATION (VALIDACIÓN)
# ─────────────────────────────────────────────────────────────────

def tune_contamination(X_train, X_val, y_val):
    candidates = [0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08]
    results = []
    print("\n  Ajuste de contamination (validación):")
    for cont in candidates:
        m = IsolationForest(n_estimators=200, contamination=cont,
                            random_state=42, n_jobs=-1)
        m.fit(X_train)
        pred = np.where(m.predict(X_val) == -1, 1, 0)
        _, _, f1, _ = precision_recall_fscore_support(
            y_val, pred, pos_label=1, average="binary", zero_division=0)
        results.append((cont, f1))
        print(f"    contamination={cont:.3f}  F1-val={f1:.4f}")
    best_cont = max(results, key=lambda r: r[1])[0]
    print(f"  → Mejor contamination: {best_cont}")
    return best_cont, results

# ─────────────────────────────────────────────────────────────────
# 6.  ENTRENAMIENTO DEL MODELO FINAL
# ─────────────────────────────────────────────────────────────────

def train_model(X_train, contamination):
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model

# ─────────────────────────────────────────────────────────────────
# 7.  EVALUACIÓN (CONJUNTO DE TESTEO)
# ─────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    raw   = model.predict(X_test)
    y_pred = np.where(raw == -1, 1, 0)

    scores = -model.score_samples(X_test)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, pos_label=1, average="binary", zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc     = auc(fpr, tpr)
    cm          = confusion_matrix(y_test, y_pred)

    print("\n  Resultados en testeo:")
    print(f"    Precisión : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    AUC-ROC   : {roc_auc:.4f}")
    tn, fp, fn, tp = cm.ravel()
    print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    return {
        "y_pred": y_pred, "scores": scores,
        "precision": precision, "recall": recall,
        "f1": f1, "auc": roc_auc,
        "fpr": fpr, "tpr": tpr, "cm": cm,
    }

# ─────────────────────────────────────────────────────────────────
# 8.  FIGURAS DE RESULTADOS
# ─────────────────────────────────────────────────────────────────

def fig_contamination_tuning(cont_results, best_cont):
    """F1-score en validación vs parámetro contamination."""
    conts = [r[0] for r in cont_results]
    f1s   = [r[1] for r in cont_results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(conts, f1s, marker="o", color=PAL_NORMAL, linewidth=2, markersize=7)
    ax.axvline(best_cont, color=PAL_ANOMALY, linestyle="--", linewidth=1.5,
               label=f"Valor óptimo = {best_cont}")
    ax.set_xlabel("Parámetro contamination")
    ax.set_ylabel("F1-Score (validación)")
    ax.set_title("Ajuste del hiperparámetro contamination — Conjunto de validación")
    ax.legend(); ax.grid(alpha=0.35)
    plt.tight_layout()
    _save(fig, "fig_contamination_tuning")


def fig_iforest_score_dist(scores, y_test):
    """Distribución de puntuaciones de anomalía normales vs anómalas."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[y_test == 0], bins=60, alpha=0.7, color=PAL_NORMAL,
            label="Normal", density=True)
    ax.hist(scores[y_test == 1], bins=30, alpha=0.75, color=PAL_ANOMALY,
            label="Anomalía", density=True)
    ax.set_xlabel("Puntuación de anomalía (normalizada)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de puntuaciones — Isolation Forest (testeo)")
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig_iforest_score_dist")


def fig_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1565C0", lw=2,
            label=f"Isolation Forest  AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Clasificador aleatorio")
    ax.fill_between(fpr, tpr, alpha=0.07, color="#1565C0")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("Curva ROC — Isolation Forest (conjunto de testeo)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_roc_curve")


def fig_confusion_matrix(cm):
    tn, fp, fn, tp = cm.ravel()
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot  = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
                       for row_v, row_p in zip(cm, cm_pct)])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Anomalía"],
                yticklabels=["Normal", "Anomalía"],
                linewidths=0.5, cbar=False)
    ax.set_title("Matriz de Confusión — Isolation Forest (testeo)")
    ax.set_xlabel("Predicción"); ax.set_ylabel("Clase real")
    plt.tight_layout()
    _save(fig, "fig_confusion_matrix")


def fig_feature_importance(model, X_train, features):
    """Importancia de variables por permutación."""
    base   = model.score_samples(X_train)
    rng    = np.random.RandomState(0)
    importances = []
    for i in range(X_train.shape[1]):
        X_p = X_train.copy()
        X_p[:, i] = rng.permutation(X_p[:, i])
        importances.append(float(np.mean(base - model.score_samples(X_p))))

    feat_map = {
        "response_time_ms": "T. respuesta total",
        "db_query_time_ms": "T. consulta BD",
        "api_response_time_ms": "T. respuesta API",
        "records_returned": "Registros retornados",
        "rows_scanned": "Filas escaneadas",
        "report_size_kb": "Tamaño reporte",
        "cpu_usage_percent": "CPU (%)",
        "memory_usage_mb": "Memoria (MB)",
        "retry_count": "Reintentos",
        "error_count": "Errores",
        "service_enc": "Servicio",
        "report_type_enc": "Tipo de reporte",
        "user_role_enc": "Rol usuario",
        "hour": "Hora del día",
    }
    labels = [feat_map.get(f, f) for f in features]
    idx    = np.argsort(importances)[::-1]
    colors = [PAL_ANOMALY if importances[i] > 0 else PAL_NORMAL for i in idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([labels[i] for i in idx], [importances[i] for i in idx], color=colors)
    ax.set_xlabel("Importancia (caída media de puntuación de normalidad)")
    ax.set_title("Importancia de variables — Isolation Forest")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    _save(fig, "fig_feature_importance")

    return [(features[i], importances[i]) for i in idx]


def fig_anomaly_profile(df):
    """
    Para cada tipo de anomalía: ratio variable_media / media_normal.
    Muestra el 'perfil' de cada anomalía respecto al comportamiento normal.
    """
    key_vars  = ["response_time_ms", "db_query_time_ms", "records_returned",
                 "report_size_kb", "cpu_usage_percent", "memory_usage_mb",
                 "retry_count", "error_count"]
    var_lbl   = {
        "response_time_ms": "T. respuesta", "db_query_time_ms": "T. consulta BD",
        "records_returned": "Registros", "report_size_kb": "Tamaño rep.",
        "cpu_usage_percent": "CPU %", "memory_usage_mb": "Memoria",
        "retry_count": "Reintentos", "error_count": "Errores",
    }
    type_lbl  = {
        "slow_database_query": "DB lenta",
        "large_report_export": "Exportación grande",
        "high_error_rate":     "Alta tasa errores",
        "high_resource_usage": "Alto uso recursos",
    }

    rs = df[df["service"] == "report-service"].copy()
    normal_means = rs[rs["anomaly_type"] == "normal"][key_vars].mean().replace(0, 1e-6)

    ratios = {}
    for at, lbl in type_lbl.items():
        s = rs[rs["anomaly_type"] == at]
        if len(s) == 0:
            continue
        ratios[lbl] = (s[key_vars].mean() / normal_means).values

    x     = np.arange(len(key_vars))
    width = 0.20
    colors = ["#E53935", "#8E24AA", "#FB8C00", "#43A047"]

    fig, ax = plt.subplots(figsize=(12, 5))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(ratios))
    for (lbl, vals), color, offset in zip(ratios.items(), colors, offsets):
        ax.bar(x + offset, vals, width, label=lbl, color=color, alpha=0.85)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Nivel normal (ratio = 1)")
    ax.set_xticks(x)
    ax.set_xticklabels([var_lbl[v] for v in key_vars], fontsize=9)
    ax.set_ylabel("Ratio respecto al valor medio normal")
    ax.set_title("Perfil de anomalías — Ratio de variables respecto al comportamiento normal (report-service)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_anomaly_profile")


def fig_anomaly_per_service(df, model, X_all):
    """Anomalías detectadas vs reales por servicio (conjunto completo)."""
    y_pred_all = np.where(model.predict(X_all) == -1, 1, 0)
    df = df.copy()
    df["predicted_anomaly"] = y_pred_all

    per_service = df.groupby("service").apply(
        lambda g: pd.Series({
            "Anomalías reales":    int(g["is_anomaly"].sum()),
            "Anomalías detectadas": int(g["predicted_anomaly"].sum()),
            "Verdaderos positivos": int(((g["is_anomaly"] == 1) & (g["predicted_anomaly"] == 1)).sum()),
        })
    ).reindex(SERVICE_ORDER)

    x = np.arange(len(per_service))
    w = 0.25
    colors_bar = [PAL_ANOMALY, "#8C4CA8", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (col, color) in enumerate(zip(per_service.columns, colors_bar)):
        ax.bar(x + (i - 1) * w, per_service[col], w,
               label=col, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(SERVICE_ORDER)
    ax.set_ylabel("Cantidad de registros")
    ax.set_title("Detección de anomalías por servicio — Isolation Forest")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_anomaly_per_service")

    return per_service

# ─────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────

def _save(fig, name):
    for folder in [FIGURES_SRC, FIGURES_LATEX]:
        for ext in ["pdf", "png"]:
            fig.savefig(os.path.join(folder, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {name}")

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("Cargando datos...")
    df = load_data()
    print_eda_summary(df)

    print("\nGenerando figuras EDA...")
    fig_anomaly_distribution(df)
    fig_response_time_boxplot(df)
    fig_correlation_heatmap(df)
    fig_temporal_patterns(df)

    print("\nPreprocesando...")
    X, y, features, scaler = preprocess(df)

    print("\nParticionando datos (70 / 15 / 15 estratificado)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"  Entrenamiento : {len(X_train):,} registros  "
          f"(anomalías: {y_train.sum():,} = {y_train.mean()*100:.2f}%)")
    print(f"  Validación    : {len(X_val):,} registros  "
          f"(anomalías: {y_val.sum():,} = {y_val.mean()*100:.2f}%)")
    print(f"  Testeo        : {len(X_test):,} registros  "
          f"(anomalías: {y_test.sum():,} = {y_test.mean()*100:.2f}%)")

    best_cont, cont_results = tune_contamination(X_train, X_val, y_val)

    print("\nEntrenando modelo final...")
    model = train_model(X_train, best_cont)

    print("\nEvaluando en testeo...")
    res = evaluate(model, X_test, y_test)

    print("\nGenerando figuras de resultados...")
    fig_contamination_tuning(cont_results, best_cont)
    fig_iforest_score_dist(res["scores"], y_test)
    fig_roc_curve(res["fpr"], res["tpr"], res["auc"])
    fig_confusion_matrix(res["cm"])
    imp = fig_feature_importance(model, X_train, features)
    fig_anomaly_profile(df)
    fig_anomaly_per_service(df, model, X)

    print("\nTop 5 variables más importantes:")
    for feat, val in imp[:5]:
        print(f"  {feat}: {val:.6f}")

    print(f"\nContamination óptimo : {best_cont}")
    print(f"Precisión            : {res['precision']:.4f}")
    print(f"Recall               : {res['recall']:.4f}")
    print(f"F1-Score             : {res['f1']:.4f}")
    print(f"AUC-ROC              : {res['auc']:.4f}")
    tn, fp, fn, tp = res["cm"].ravel()
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    print("\nAnálisis completado.")
    return res, best_cont, imp, y_train, y_val, y_test


if __name__ == "__main__":
    main()
