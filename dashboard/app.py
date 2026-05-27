import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from typing import Any

PLOTLY_LAYOUT: dict[str, Any] = {
    "template": "plotly_white",
    "margin": {"l": 20, "r": 20, "t": 50, "b": 20},
}

st.set_page_config(
    page_title="NASA CMAPSS — Maintenance Prédictive",
    page_icon="⚙️",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0A0D14;
}
[data-testid="stHeader"] {
    background-color: #0A0D14;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', 'Inter', sans-serif;
    color: #F1F5F9;
}
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"]:hover {
    border-color: #22D3EE;
}
[data-testid="stMetricValue"] {
    color: #22D3EE;
    font-family: 'DM Mono', monospace;
    font-size: 28px !important;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
h1 {
    font-family: 'DM Mono', monospace !important;
    color: #F1F5F9 !important;
    font-size: 28px !important;
    padding-bottom: 16px;
}
h2 {
    font-family: 'DM Mono', monospace !important;
    color: #22D3EE !important;
    font-size: 16px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
h3 {
    color: #F1F5F9 !important;
    font-size: 15px !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #1E293B;
    border-radius: 8px;
}
[data-testid="stSelectbox"] > div > div {
    background: #111827;
    border-color: #1E293B;
    color: #F1F5F9;
}
[data-testid="stAlert"] {
    border-radius: 8px;
}
hr {
    border-color: #1E293B;
}
[data-testid="stPlotlyChart"] {
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 4px;
    background: #111827;
}
</style>
""", unsafe_allow_html=True)


DATABASE_PATH = Path("database/cmapss.db")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#0A0D14",
    font=dict(family="DM Mono, monospace", color="#94A3B8", size=11),
    title_font=dict(family="DM Mono, monospace", color="#F1F5F9", size=13),
    xaxis=dict(
        gridcolor="#1E293B",
        linecolor="#1E293B",
        tickcolor="#1E293B",
        tickfont=dict(color="#94A3B8", size=10),
    ),
    yaxis=dict(
        gridcolor="#1E293B",
        linecolor="#1E293B",
        tickcolor="#1E293B",
        tickfont=dict(color="#94A3B8", size=10),
    ),
    legend=dict(
        bgcolor="#111827",
        bordercolor="#1E293B",
        borderwidth=1,
        font=dict(color="#94A3B8", size=10),
    ),
    margin=dict(l=48, r=24, t=48, b=36),
)

RISK_COLORS = {
    "Critique": "#E24B4A",
    "À surveiller": "#EF9F27",
    "Stable": "#22D3EE",
}

RISK_LABELS = {
    "HIGH": "Critique",
    "MEDIUM": "À surveiller",
    "LOW": "Stable",
}

RISK_ORDER = {
    "Critique": 0,
    "À surveiller": 1,
    "Stable": 2,
}


SENSOR_NAMES = {
    "sensor_2": "T24 — Température totale sortie compresseur basse pression | Capteur 2",
    "sensor_3": "T30 — Température totale sortie compresseur haute pression | Capteur 3",
    "sensor_4": "T50 — Température totale sortie turbine basse pression | Capteur 4",
    "sensor_7": "P30 — Pression totale sortie compresseur haute pression | Capteur 7",
    "sensor_8": "Nf — Vitesse physique du fan | Capteur 8",
    "sensor_9": "Nc — Vitesse physique du cœur moteur | Capteur 9",
    "sensor_11": "Ps30 — Pression statique sortie compresseur haute pression | Capteur 11",
    "sensor_12": "phi — Ratio fuel flow / Ps30 | Capteur 12",
    "sensor_13": "NRf — Vitesse fan corrigée | Capteur 13",
    "sensor_14": "NRc — Vitesse cœur corrigée | Capteur 14",
    "sensor_15": "BPR — Bypass Ratio | Capteur 15",
    "sensor_17": "htBleed — Enthalpie de prélèvement | Capteur 17",
    "sensor_20": "W31 — Débit d’air HPT coolant | Capteur 20",
    "sensor_21": "W32 — Débit d’air LPT coolant | Capteur 21",
}


def format_sensor_name(sensor: str) -> str:
    return SENSOR_NAMES.get(sensor, sensor.replace("sensor_", "Capteur "))


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required_files = [
        PROCESSED_DIR / "test_predictions.csv",
        PROCESSED_DIR / "test_features.csv",
        MODELS_DIR / "metrics.json",
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        st.error("Fichiers manquants. Lancez d'abord le pipeline de modélisation.")
        st.write(missing_files)
        st.stop()

    if DATABASE_PATH.exists():
        conn = sqlite3.connect(DATABASE_PATH)

        test_pred = pd.read_sql_query(
            """
            SELECT
                unit_id,
                last_cycle,
                true_rul AS true_RUL,
                predicted_rul AS predicted_RUL,
                absolute_error,
                risk_level
            FROM predictions
            ORDER BY
                CASE risk_level
                    WHEN 'HIGH' THEN 1
                    WHEN 'MEDIUM' THEN 2
                    WHEN 'LOW' THEN 3
                END,
                predicted_rul ASC
            """,
            conn,
        )

        test_features = pd.read_sql_query(
            "SELECT * FROM sensor_readings",
            conn,
        )

        conn.close()

    else:
        test_pred = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
        test_features = pd.read_csv(PROCESSED_DIR / "test_features.csv")

    with open(MODELS_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    test_pred["risk_label"] = test_pred["risk_level"].map(RISK_LABELS)

    return test_pred, test_features, metrics


def build_model_comparison(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Évaluation": "Validation interne",
                "Modèle": "Ridge baseline",
                "RMSE": metrics["validation_all_cycles"]["ridge_baseline"]["rmse"],
                "MAE": metrics["validation_all_cycles"]["ridge_baseline"]["mae"],
            },
            {
                "Évaluation": "Validation interne",
                "Modèle": "Random Forest",
                "RMSE": metrics["validation_all_cycles"]["random_forest"]["rmse"],
                "MAE": metrics["validation_all_cycles"]["random_forest"]["mae"],
            },
        ]
    )


def prepare_display_table(test_pred: pd.DataFrame) -> pd.DataFrame:
    df_display = test_pred[
        [
            "unit_id",
            "last_cycle",
            "true_RUL",
            "predicted_RUL",
            "absolute_error",
            "risk_label",
        ]
    ].copy()

    df_display["risk_order"] = df_display["risk_label"].map(RISK_ORDER)
    df_display = df_display.sort_values(["risk_order", "predicted_RUL"])
    df_display = df_display.drop(columns="risk_order")

    df_display.columns = [
        "Moteur",
        "Dernier cycle observé",
        "RUL réel",
        "RUL prédit",
        "Erreur absolue",
        "Risque",
    ]

    return df_display


def get_sensor_cols(test_features: pd.DataFrame) -> list[str]:
    return [
        col
        for col in test_features.columns
        if col.startswith("sensor_")
        and "rollmean" not in col
        and "delta" not in col
    ]


def get_action_message(risk_label: str) -> str:
    if risk_label == "Critique":
        return "Maintenance prioritaire recommandée : le moteur présente un RUL prédit très faible."
    if risk_label == "À surveiller":
        return "Maintenance à planifier : le moteur n'est pas critique, mais doit être suivi de près."
    return "Surveillance normale : le moteur ne présente pas de signal prioritaire selon le RUL prédit."


def build_risk_summary(test_pred: pd.DataFrame) -> pd.DataFrame:
    risk_summary = (
        test_pred["risk_label"]
        .value_counts()
        .rename_axis("Risque")
        .reset_index(name="Nombre de moteurs")
    )

    risk_summary["risk_order"] = risk_summary["Risque"].map(RISK_ORDER)
    risk_summary = risk_summary.sort_values("risk_order").drop(columns="risk_order")

    return risk_summary


test_pred, test_features, metrics = load_data()

test_metrics = metrics["test_last_cycle"]["random_forest"]
rmse = test_metrics["rmse"]
mae = test_metrics["mae"]
nasa = test_metrics["nasa_score"]

n_critical = int((test_pred["risk_label"] == "Critique").sum())
n_watch = int((test_pred["risk_label"] == "À surveiller").sum())
n_stable = int((test_pred["risk_label"] == "Stable").sum())
mean_predicted_rul = float(test_pred["predicted_RUL"].mean())


st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
  <span style="font-family:monospace;font-size:11px;color:#94A3B8;letter-spacing:0.12em;">
    NASA CMAPSS FD001 · TURBOFAN FLEET MONITOR
  </span>
  <span style="background:rgba(34,211,238,0.1);color:#22D3EE;border:1px solid rgba(34,211,238,0.3);
    border-radius:999px;padding:3px 10px;font-size:11px;font-family:monospace;">
    ● BATCH INFERENCE
  </span>
</div>
""", unsafe_allow_html=True)

st.title("NASA CMAPSS — Tableau de bord de Maintenance Prédictive")

st.markdown(
    """
Ce dashboard transforme les prédictions de **durée de vie résiduelle (RUL)** en outil d'aide à la décision pour la maintenance industrielle.

**Question métier : quels moteurs doivent être surveillés ou maintenus en priorité ?**

Le modèle prédit le nombre de cycles restants avant panne à partir des signaux capteurs du dataset **NASA CMAPSS FD001**, issu de la compétition NASA/PHM'08.
"""
)

st.divider()


st.header("1. Vue opérationnelle — Priorisation de maintenance")

st.markdown(
    """
Les moteurs sont classés selon leur **RUL prédit**.
Plus le RUL prédit est faible, plus le moteur doit être traité en priorité.
"""
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Moteurs critiques",
    n_critical,
    help="RUL prédit ≤ 30 cycles. Ces moteurs doivent faire l'objet d'une intervention de maintenance prioritaire.",
)

col2.metric(
    "Moteurs à surveiller",
    n_watch,
    help="RUL prédit entre 30 et 60 cycles. Une maintenance est à planifier dans un délai raisonnable.",
)

col3.metric(
    "Moteurs stables",
    n_stable,
    help="RUL prédit > 60 cycles. Ces moteurs ne présentent pas de signal d'alerte immédiat.",
)

col4.metric(
    "RUL moyen prédit",
    f"{mean_predicted_rul:.1f} cycles",
    help="Durée de vie résiduelle moyenne prédite sur l'ensemble des 100 moteurs du test set NASA.",
)

st.info(
    "**Règle de décision :** Critique ≤ 30 cycles · À surveiller ≤ 60 cycles · Stable > 60 cycles"
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    "<p style='font-family:monospace;font-size:11px;color:#94A3B8;letter-spacing:0.1em;'>"
    "TRAJECTOIRES CAPTEURS · Début de vie → Fin de vie</p>",
    unsafe_allow_html=True,
)

col_chart, col_table = st.columns([3, 2])

with col_chart:
    sensor_cols = get_sensor_cols(test_features)

    if not sensor_cols:
        st.warning("Aucun capteur exploitable n'a été trouvé dans les features.")
        st.stop()

    default_global_sensor = "sensor_2" if "sensor_2" in sensor_cols else sensor_cols[0]

    selected_global_sensor = st.selectbox(
        "Capteur affiché sur la vue flotte",
        options=sensor_cols,
        index=sensor_cols.index(default_global_sensor),
        format_func=format_sensor_name,
        key="global_sensor_selector",
    )

    sensor = selected_global_sensor

    fig_deg = go.Figure()

    fig_deg.add_vrect(
        x0=0,
        x1=30,
        fillcolor="rgba(226,75,74,0.20)",
        line_width=0,
        layer="below",
    )

    fig_deg.add_vrect(
        x0=30,
        x1=60,
        fillcolor="rgba(239,159,39,0.13)",
        line_width=0,
        layer="below",
    )

    fig_deg.add_vline(
        x=30,
        line_dash="dot",
        line_color="#E24B4A",
        line_width=1.3,
        opacity=0.85,
    )

    fig_deg.add_vline(
        x=60,
        line_dash="dot",
        line_color="#EF9F27",
        line_width=1.3,
        opacity=0.85,
    )

    fig_deg.add_annotation(
        x=15,
        y=0.93,
        xref="x",
        yref="paper",
        text="<b>CRITIQUE</b><br>≤ 30 cycles",
        showarrow=False,
        align="center",
        font=dict(color="#FF6B81", size=11, family="DM Mono, monospace"),
        bgcolor="rgba(10,13,20,0.82)",
        bordercolor="rgba(226,75,74,0.75)",
        borderwidth=1,
        borderpad=5,
    )

    fig_deg.add_annotation(
        x=45,
        y=0.93,
        xref="x",
        yref="paper",
        text="<b>À SURVEILLER</b><br>≤ 60 cycles",
        showarrow=False,
        align="center",
        font=dict(color="#F5B041", size=11, family="DM Mono, monospace"),
        bgcolor="rgba(10,13,20,0.82)",
        bordercolor="rgba(239,159,39,0.75)",
        borderwidth=1,
        borderpad=5,
    )

    if len(test_features) > 0:
        for uid in test_pred["unit_id"].unique()[:30]:
            unit_data_deg = test_features[test_features["unit_id"] == uid].copy()
            if len(unit_data_deg) < 3:
                continue
            risk = test_pred[test_pred["unit_id"] == uid]["risk_label"].values[0]
            s = unit_data_deg[sensor].values
            s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
            x_rul = list(range(len(s_norm) - 1, -1, -1))
            color = RISK_COLORS.get(risk, "#22D3EE")
            fig_deg.add_trace(go.Scatter(
                x=x_rul,
                y=s_norm,
                mode="lines",
                line=dict(color=color, width=0.8),
                opacity=0.10,
                showlegend=False,
                hoverinfo="skip",
            ))

    critical_units = test_pred[test_pred["risk_label"] == "Critique"].sort_values("predicted_RUL")
    if len(critical_units) > 0:
        top_unit = critical_units.iloc[0]
        uid = top_unit["unit_id"]
        rul_pred = top_unit["predicted_RUL"]
        unit_data_crit = test_features[test_features["unit_id"] == uid].copy()
        if len(unit_data_crit) > 0 and sensor in unit_data_crit.columns:
            s = unit_data_crit[sensor].values
            s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
            x_rul = list(range(len(s_norm) - 1, -1, -1))
            marker_y = float(s_norm[-1]) if len(s_norm) > 0 else 0.5
            fig_deg.add_trace(go.Scatter(
                x=x_rul,
                y=s_norm,
                mode="lines",
                line=dict(color="#22D3EE", width=2.6),
                name=f"#{uid:03d} · CRITIQUE",
                showlegend=False,
            ))
            fig_deg.add_trace(go.Scatter(
                x=[rul_pred],
                y=[marker_y],
                mode="markers+text",
                marker=dict(
                    color="#22D3EE",
                    size=11,
                    line=dict(color="#0A0D14", width=2),
                ),
                text=[f"RUL={rul_pred:.0f} · #{uid:03d}"],
                textposition="top right",
                textfont=dict(color="#22D3EE", size=9, family="DM Mono, monospace"),
                showlegend=False,
            ))

    fig_deg.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Trajectoires capteurs normalisées · {format_sensor_name(sensor)}",
        xaxis_title="RUL prédit (cycles restants avant panne)",
        yaxis_title="Signal capteur normalisé",
        height=400,
        showlegend=False,
    )
    fig_deg.update_xaxes(range=[165, -5])
    fig_deg.update_yaxes(range=[-0.08, 1.08])
    fig_deg.add_annotation(
        x=0,
        y=-0.13,
        xref="paper",
        yref="paper",
        text="Début de vie moteur",
        showarrow=False,
        font=dict(color="#94A3B8", size=9, family="DM Mono, monospace"),
    )
    fig_deg.add_annotation(
        x=1,
        y=-0.13,
        xref="paper",
        yref="paper",
        text="Fin de vie moteur",
        showarrow=False,
        font=dict(color="#94A3B8", size=9, family="DM Mono, monospace"),
    )

    st.plotly_chart(fig_deg, use_container_width=True)
    st.caption(
        "Visualisation des trajectoires capteurs associées aux seuils de priorisation maintenance. "
    "La courbe cyan met en évidence le moteur le plus urgent selon le RUL prédit."
    )

with col_table:
    st.markdown(
        "<p style='font-family:monospace;font-size:11px;color:#94A3B8;"
        "letter-spacing:0.1em;margin-bottom:12px;'>TOP RISQUE · MOTEURS</p>",
        unsafe_allow_html=True,
    )

    _top = test_pred.copy()
    _top["_risk_order"] = _top["risk_label"].map(RISK_ORDER)
    top_engines = _top.sort_values(["_risk_order", "predicted_RUL"]).head(8)[
        ["unit_id", "predicted_RUL", "risk_label"]
    ]

    for _, row in top_engines.iterrows():
        color = RISK_COLORS.get(row["risk_label"], "#22D3EE")
        label = row["risk_label"].upper()[:8]
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
            padding:8px 12px;border-bottom:1px solid #1E293B;font-family:monospace;font-size:12px;">
          <span style="color:#94A3B8;">#{row['unit_id']:03d}</span>
          <span style="color:{color};font-weight:500;">{row['predicted_RUL']:.0f} cyc</span>
          <span style="background:{color}22;color:{color};border:1px solid {color}55;
            border-radius:4px;padding:2px 8px;font-size:10px;">{label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(
        "RMSE test",
        f"{rmse:.2f} cyc",
        help="Erreur quadratique moyenne sur les 100 moteurs test NASA. Exprimée en cycles restants.",
    )
    mc2.metric(
        "MAE test",
        f"{mae:.2f} cyc",
        help="Erreur absolue moyenne sur les 100 moteurs test NASA. Le modèle se trompe en moyenne de cette valeur en cycles.",
    )
    mc3.metric(
        "NASA Score",
        f"{nasa:,.0f}",
        help="Score asymétrique PHM'08. Plus il est faible, meilleur est le modèle. Les surestimations du RUL sont davantage pénalisées car elles peuvent conduire à une panne non anticipée.",
    )

risk_summary = build_risk_summary(test_pred)

fig_risk = px.bar(
    risk_summary,
    x="Risque",
    y="Nombre de moteurs",
    color="Risque",
    color_discrete_map=RISK_COLORS,
    title="Répartition des moteurs par niveau de risque",
    text="Nombre de moteurs",
)
fig_risk.update_layout(**PLOTLY_LAYOUT, showlegend=False)
fig_risk.update_traces(textposition="outside")
st.plotly_chart(fig_risk, use_container_width=True)

st.subheader("Moteurs à prioriser")

if DATABASE_PATH.exists():
    st.caption("Vue de priorisation maintenance — moteurs classés du plus critique au plus stable.")
else:
    st.caption("Vue de priorisation maintenance — moteurs classés du plus critique au plus stable.")

priority_table = prepare_display_table(test_pred)
st.dataframe(priority_table, use_container_width=True, hide_index=True)

st.divider()


st.header("2. Analyse détaillée d'un moteur")

st.markdown(
    """
Sélectionnez un moteur pour visualiser son RUL prédit, son niveau de risque et l'évolution de ses capteurs.

Le **RUL réel** est affiché car le dataset NASA fournit les valeurs de référence pour le test set, ce qui permet d'évaluer la qualité des prédictions.
Dans un cas d'usage industriel réel, seul le RUL prédit serait disponible.
"""
)

selected_unit = st.selectbox(
    "Sélectionner un moteur",
    options=sorted(test_pred["unit_id"].unique()),
    format_func=lambda x: f"Moteur {x}",
)

unit_data = test_features[test_features["unit_id"] == selected_unit].copy()
unit_pred = test_pred[test_pred["unit_id"] == selected_unit].iloc[0]
unit_risk_label = unit_pred["risk_label"]

col_a, col_b, col_c, col_d = st.columns(4)

col_a.metric(
    "RUL prédit",
    f"{unit_pred['predicted_RUL']:.0f} cycles",
    help="Nombre de cycles restants avant panne estimé par le modèle Random Forest.",
)
col_b.metric(
    "RUL réel",
    f"{unit_pred['true_RUL']:.0f} cycles",
    help="Valeur réelle fournie par le dataset NASA. Disponible uniquement pour évaluation — ne serait pas connu en production réelle.",
)
col_c.metric(
    "Erreur de prédiction",
    f"{unit_pred['absolute_error']:.0f} cycles",
    help="Écart absolu entre le RUL prédit et le RUL réel. Plus cette valeur est faible, meilleure est la prédiction.",
)
col_d.metric(
    "Niveau de risque",
    unit_risk_label,
    help="Critique : RUL ≤ 30 cycles. À surveiller : RUL ≤ 60 cycles. Stable : RUL > 60 cycles.",
)

if unit_risk_label == "Critique":
    st.error(get_action_message(unit_risk_label))
elif unit_risk_label == "À surveiller":
    st.warning(get_action_message(unit_risk_label))
else:
    st.success(get_action_message(unit_risk_label))

sensor_cols = get_sensor_cols(test_features)

if not sensor_cols:
    st.warning("Aucun capteur exploitable n'a été trouvé dans les features.")
    st.stop()

selected_sensor = st.selectbox(
    "Sélectionner un capteur à visualiser",
    options=sensor_cols,
    format_func=format_sensor_name,
    key="detail_sensor_selector",
)

rollmean_col = f"{selected_sensor}_rollmean_5"

fig_sensor = go.Figure()

fig_sensor.add_trace(
    go.Scatter(
        x=unit_data["cycle"],
        y=unit_data[selected_sensor],
        mode="lines",
        name="Signal brut",
        line=dict(color="#334155", width=1),
        opacity=0.7,
    )
)

if rollmean_col in unit_data.columns:
    fig_sensor.add_trace(
        go.Scatter(
            x=unit_data["cycle"],
            y=unit_data[rollmean_col],
            mode="lines",
            name="Moyenne glissante (5 cycles)",
            line=dict(color="#22D3EE", width=2),
        )
    )

fig_sensor.update_layout(
    **PLOTLY_LAYOUT,
    title=f"Évolution de {format_sensor_name(selected_sensor)} — Moteur {selected_unit}",
    xaxis_title="Cycle",
    yaxis_title="Valeur du capteur",
    height=340,
)
fig_sensor.update_layout(legend=dict(orientation="h"))

st.plotly_chart(fig_sensor, use_container_width=True)

st.caption(
    """
Le signal brut montre les mesures cycle par cycle.
La moyenne glissante sur 5 cycles lisse les fluctuations locales liées au bruit de mesure — inhérent au dataset NASA CMAPSS — et rend la tendance de dégradation plus lisible pour le modèle.
"""
)

st.divider()


st.header("3. Performance du modèle — Fiabilité des prédictions")

st.markdown(
    """
Le modèle est évalué sur les **100 moteurs test** du dataset NASA CMAPSS FD001.
L'évaluation porte sur le **dernier cycle observé** de chaque moteur : c'est le scénario opérationnel principal,
celui où l'on demande au modèle de prédire combien de cycles il reste avant la panne.
"""
)

col_m1, col_m2, col_m3 = st.columns(3)

col_m1.metric(
    "RMSE — test NASA",
    f"{rmse:.1f} cycles",
    help="Racine de l'erreur quadratique moyenne. Pénalise davantage les grandes erreurs. "
         "Exprimée en cycles : le modèle se trompe en moyenne de cette valeur, avec plus de poids sur les erreurs importantes.",
)

col_m2.metric(
    "MAE — test NASA",
    f"{mae:.1f} cycles",
    help="Erreur absolue moyenne. Directement interprétable : le modèle se trompe en moyenne de cette valeur en cycles, "
         "sans surpondérer les grandes erreurs.",
)

col_m3.metric(
    "Score NASA (PHM'08)",
    f"{nasa:.0f}",
    help="Métrique asymétrique définie par Saxena et al. (2008). "
         "Pénalise davantage les surestimations du RUL, car elles peuvent retarder une intervention et conduire à une panne non anticipée. "
         "Plus ce score est faible, meilleur est le modèle. Un score de 0 correspondrait à des prédictions parfaites.",
)

col_left, col_right = st.columns(2)

with col_left:
    fig_scatter = px.scatter(
        test_pred,
        x="true_RUL",
        y="predicted_RUL",
        color="risk_label",
        color_discrete_map=RISK_COLORS,
        hover_data={
            "unit_id": True,
            "absolute_error": True,
            "risk_label": True,
        },
        title="RUL prédit vs RUL réel — 100 moteurs test",
        labels={
            "true_RUL": "RUL réel (cycles)",
            "predicted_RUL": "RUL prédit (cycles)",
            "risk_label": "Risque",
            "absolute_error": "Erreur absolue",
            "unit_id": "Moteur",
        },
    )

    max_rul = max(
        float(test_pred["true_RUL"].max()),
        float(test_pred["predicted_RUL"].max()),
    )

    fig_scatter.add_shape(
        type="line",
        x0=0, y0=0,
        x1=max_rul, y1=max_rul,
        line=dict(color="#334155", dash="dash"),
    )

    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=340)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "Chaque point représente un moteur test. La diagonale pointillée correspond à une prédiction parfaite. "
        "Les points au-dessus de la diagonale indiquent une surestimation du RUL (risque métier élevé) ; "
        "les points en dessous une sous-estimation (maintenance anticipée)."
    )

with col_right:
    fig_error = px.histogram(
        test_pred,
        x="absolute_error",
        nbins=30,
        color="risk_label",
        color_discrete_map=RISK_COLORS,
        title="Distribution des erreurs absolues par niveau de risque",
        labels={
            "absolute_error": "Erreur absolue (cycles)",
            "risk_label": "Risque",
        },
    )

    fig_error.update_layout(**PLOTLY_LAYOUT, height=340)
    st.plotly_chart(fig_error, use_container_width=True)
    st.caption(
        "Distribution des erreurs absolues (|RUL réel − RUL prédit|) pour l'ensemble des moteurs test. "
        "Une distribution concentrée vers 0 indique un modèle précis. "
        "Les erreurs des moteurs critiques sont particulièrement importantes à surveiller."
    )

st.divider()


st.header("4. Comparaison avec une baseline")

st.markdown(
    """
Le **Random Forest** est comparé à une **Ridge Regression**, utilisée comme baseline simple et interprétable.

Sur la validation interne, le Random Forest réduit les erreurs de prédiction par rapport à la baseline.
Cette amélioration se traduit par une RMSE et une MAE plus faibles, deux métriques qui mesurent l'écart entre le RUL réel et le RUL prédit.

**Note sur l'évaluation :** les métriques ci-dessous sont calculées sur tous les cycles du split de validation interne (80 moteurs train / 20 moteurs validation).
Elles ne sont pas directement comparables aux métriques du test set NASA, qui portent uniquement sur le dernier cycle observé de chaque moteur.
"""
)

comparison_df = build_model_comparison(metrics)

st.dataframe(comparison_df, use_container_width=True, hide_index=True)

fig_comparison = px.bar(
    comparison_df,
    x="Évaluation",
    y="RMSE",
    color="Modèle",
    color_discrete_map={
        "Ridge baseline": "#334155",
        "Random Forest": "#22D3EE",
    },
    barmode="group",
    title="Comparaison RMSE — Ridge baseline vs Random Forest (validation interne)",
    labels={"RMSE": "RMSE (cycles)"},
)

fig_comparison.update_layout(**PLOTLY_LAYOUT, height=300)
st.plotly_chart(fig_comparison, use_container_width=True)

st.divider()

st.markdown(
    f"""
### Résumé

Ce dashboard répond à trois questions :

1. **Quels moteurs sont prioritaires pour la maintenance ?**
   → Vue opérationnelle, courbe de dégradation et table de priorisation.

2. **Quel est l'état détaillé d'un moteur donné ?**
   → Analyse individuelle avec RUL prédit, RUL réel, erreur de prédiction, niveau de risque et courbes capteurs.

3. **Peut-on faire confiance aux prédictions ?**
   → RMSE de **{rmse:.1f} cycles** et MAE de **{mae:.1f} cycles** sur le test set NASA CMAPSS FD001,
   pour une première approche Random Forest sans optimisation avancée des hyperparamètres.
"""
)

st.markdown(f"""
<div style="font-family:monospace;font-size:11px;color:#94A3B8;
    border-top:1px solid #1E293B;padding-top:16px;margin-top:8px;
    display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">
  <span>NASA CMAPSS FD001 · Random Forest · n_estimators=300</span>
  <span>RMSE {rmse:.2f} cyc · MAE {mae:.2f} cyc · NASA Score {nasa:,.0f}</span>
  <span>100 moteurs test · 21 capteurs → 24 features</span>
</div>
""", unsafe_allow_html=True)