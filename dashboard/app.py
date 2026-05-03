import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="NASA CMAPSS — Maintenance Prédictive",
    page_icon="⚙️",
    layout="wide",
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

RISK_COLORS = {
    "Critique": "#E24B4A",
    "À surveiller": "#EF9F27",
    "Stable": "#378ADD",
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
                "Évaluation": "Tous les cycles — validation",
                "Modèle": "Ridge baseline",
                "RMSE": metrics["validation_all_cycles"]["ridge_baseline"]["rmse"],
                "MAE": metrics["validation_all_cycles"]["ridge_baseline"]["mae"],
                "Score NASA": metrics["validation_all_cycles"]["ridge_baseline"]["nasa_score"],
            },
            {
                "Évaluation": "Tous les cycles — validation",
                "Modèle": "Random Forest",
                "RMSE": metrics["validation_all_cycles"]["random_forest"]["rmse"],
                "MAE": metrics["validation_all_cycles"]["random_forest"]["mae"],
                "Score NASA": metrics["validation_all_cycles"]["random_forest"]["nasa_score"],
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
    help="RUL prédit ≤ 30 cycles — Maintenance prioritaire",
)

col2.metric(
    "Moteurs à surveiller",
    n_watch,
    help="RUL prédit ≤ 60 cycles — Maintenance à planifier",
)

col3.metric(
    "Moteurs stables",
    n_stable,
    help="RUL prédit > 60 cycles — Surveillance normale",
)

col4.metric(
    "RUL moyen prédit",
    f"{mean_predicted_rul:.1f} cycles",
)

st.info(
    "**Règle de décision :** Critique ≤ 30 cycles · À surveiller ≤ 60 cycles · Stable > 60 cycles"
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

fig_risk.update_layout(showlegend=False)
fig_risk.update_traces(textposition="outside")

st.plotly_chart(fig_risk, use_container_width=True)

priority_table = prepare_display_table(test_pred)

st.subheader("Moteurs à prioriser")
st.caption("Table triée par niveau de risque puis par RUL prédit croissant.")

st.dataframe(priority_table, use_container_width=True, hide_index=True)

st.divider()

st.header("2. Analyse détaillée d'un moteur")

st.markdown(
    """
Sélectionnez un moteur pour visualiser son RUL prédit, son niveau de risque et l'évolution de ses capteurs.

Le **RUL réel** est affiché car le dataset NASA fournit les valeurs de référence, ce qui permet d'évaluer les prédictions.
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

col_a.metric("RUL prédit", f"{unit_pred['predicted_RUL']:.0f} cycles")
col_b.metric("RUL réel", f"{unit_pred['true_RUL']:.0f} cycles")
col_c.metric("Erreur", f"{unit_pred['absolute_error']:.0f} cycles")
col_d.metric("Niveau de risque", unit_risk_label)

if unit_risk_label == "Critique":
    st.error(get_action_message(unit_risk_label))
elif unit_risk_label == "À surveiller":
    st.warning(get_action_message(unit_risk_label))
else:
    st.success(get_action_message(unit_risk_label))

sensor_cols = get_sensor_cols(test_features)

selected_sensor = st.selectbox(
    "Sélectionner un capteur à visualiser",
    options=sensor_cols,
    format_func=lambda x: x.replace("sensor_", "Capteur "),
)

rollmean_col = f"{selected_sensor}_rollmean_5"

fig_sensor = go.Figure()

fig_sensor.add_trace(
    go.Scatter(
        x=unit_data["cycle"],
        y=unit_data[selected_sensor],
        mode="lines",
        name="Signal brut",
        line=dict(color="#888780", width=1),
        opacity=0.5,
    )
)

if rollmean_col in unit_data.columns:
    fig_sensor.add_trace(
        go.Scatter(
            x=unit_data["cycle"],
            y=unit_data[rollmean_col],
            mode="lines",
            name="Moyenne glissante (5 cycles)",
            line=dict(color="#378ADD", width=2),
        )
    )

fig_sensor.update_layout(
    title=f"Évolution de {selected_sensor} — Moteur {selected_unit}",
    xaxis_title="Cycle",
    yaxis_title="Valeur du capteur",
    legend=dict(orientation="h"),
)

st.plotly_chart(fig_sensor, use_container_width=True)

st.caption(
    """
Le signal brut montre les mesures cycle par cycle. 
La moyenne glissante sur 5 cycles lisse les fluctuations locales pour rendre la tendance plus lisible.
"""
)

st.divider()

st.header("3. Performance du modèle — Fiabilité des prédictions")

st.markdown(
    """
Le modèle est évalué sur les 100 moteurs test.  
L'évaluation porte sur le **dernier cycle observé** de chaque moteur : c'est le scénario opérationnel principal.
"""
)

col_m1, col_m2, col_m3 = st.columns(3)

col_m1.metric(
    "RMSE",
    f"{rmse:.1f} cycles",
    help="Erreur quadratique moyenne. Plus elle est faible, meilleure est la prédiction.",
)

col_m2.metric(
    "MAE",
    f"{mae:.1f} cycles",
    help="Erreur absolue moyenne en cycles.",
)

col_m3.metric(
    "Score NASA",
    f"{nasa:.0f}",
    help="Score asymétrique PHM'08 : les erreurs dangereuses sont davantage pénalisées.",
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
        title="RUL prédit vs RUL réel",
        labels={
            "true_RUL": "RUL réel",
            "predicted_RUL": "RUL prédit",
            "risk_label": "Risque",
            "absolute_error": "Erreur absolue",
        },
    )

    max_rul = max(
        float(test_pred["true_RUL"].max()),
        float(test_pred["predicted_RUL"].max()),
    )

    fig_scatter.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_rul,
        y1=max_rul,
        line=dict(color="gray", dash="dash"),
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    fig_error = px.histogram(
        test_pred,
        x="absolute_error",
        nbins=30,
        color="risk_label",
        color_discrete_map=RISK_COLORS,
        title="Distribution des erreurs absolues",
        labels={
            "absolute_error": "Erreur absolue (cycles)",
            "risk_label": "Risque",
        },
    )

    st.plotly_chart(fig_error, use_container_width=True)

st.divider()

st.header("4. Comparaison avec une baseline")

st.markdown(
    """
Le **Random Forest** est comparé à une **Ridge Regression**, utilisée comme baseline simple et interprétable.

Sur la validation interne, le Random Forest améliore la RMSE et la MAE par rapport à la baseline.  
Le score NASA reste cependant plus élevé, ce qui indique que certaines erreurs sont davantage pénalisées par la métrique asymétrique PHM'08.
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
        "Ridge baseline": "#888780",
        "Random Forest": "#378ADD",
    },
    barmode="group",
    title="Comparaison RMSE — Ridge baseline vs Random Forest",
    labels={"RMSE": "RMSE (cycles)"},
)

st.plotly_chart(fig_comparison, use_container_width=True)

st.divider()

st.markdown(
    f"""
### Résumé

Ce dashboard répond à trois questions :

1. **Quels moteurs sont prioritaires pour la maintenance ?**  
   → Vue opérationnelle, graphique des risques et table de priorisation.

2. **Quel est l'état détaillé d'un moteur donné ?**  
   → Analyse individuelle avec RUL prédit, niveau de risque et courbes capteurs.

3. **Quelle est la fiabilité du modèle de prédiction RUL ?**  
   → RMSE de **{rmse:.1f} cycles** sur le test set NASA CMAPSS FD001, pour une première approche Random Forest sans optimisation avancée.
"""
)