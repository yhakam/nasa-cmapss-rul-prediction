import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="NASA CMAPSS — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

RISK_COLORS = {
    "HIGH": "#E24B4A",
    "MEDIUM": "#EF9F27",
    "LOW": "#378ADD",
}

RISK_LABELS = {
    "HIGH": "Critique",
    "MEDIUM": "Moyen",
    "LOW": "Faible",
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

    return test_pred, test_features, metrics


def build_model_comparison(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Évaluation": "Tous les cycles validation",
                "Modèle": "Ridge baseline",
                "RMSE": metrics["validation_all_cycles"]["ridge_baseline"]["rmse"],
                "MAE": metrics["validation_all_cycles"]["ridge_baseline"]["mae"],
                "Score NASA": metrics["validation_all_cycles"]["ridge_baseline"]["nasa_score"],
            },
            {
                "Évaluation": "Tous les cycles validation",
                "Modèle": "Random Forest",
                "RMSE": metrics["validation_all_cycles"]["random_forest"]["rmse"],
                "MAE": metrics["validation_all_cycles"]["random_forest"]["mae"],
                "Score NASA": metrics["validation_all_cycles"]["random_forest"]["nasa_score"],
            },
            {
                "Évaluation": "Dernier cycle par moteur",
                "Modèle": "Ridge baseline",
                "RMSE": metrics["validation_last_cycle"]["ridge_baseline"]["rmse"],
                "MAE": metrics["validation_last_cycle"]["ridge_baseline"]["mae"],
                "Score NASA": metrics["validation_last_cycle"]["ridge_baseline"]["nasa_score"],
            },
            {
                "Évaluation": "Dernier cycle par moteur",
                "Modèle": "Random Forest",
                "RMSE": metrics["validation_last_cycle"]["random_forest"]["rmse"],
                "MAE": metrics["validation_last_cycle"]["random_forest"]["mae"],
                "Score NASA": metrics["validation_last_cycle"]["random_forest"]["nasa_score"],
            },
        ]
    )


def prepare_display_table(test_pred: pd.DataFrame) -> pd.DataFrame:
    df_display = test_pred[
        ["unit_id", "last_cycle", "true_RUL", "predicted_RUL", "absolute_error", "risk_level"]
    ].copy()

    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    df_display["risk_order"] = df_display["risk_level"].map(risk_order)
    df_display = df_display.sort_values(["risk_order", "predicted_RUL"])

    df_display["risk_level"] = df_display["risk_level"].map(RISK_LABELS)
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


test_pred, test_features, metrics = load_data()

test_metrics = metrics["test_last_cycle"]["random_forest"]

rmse = test_metrics["rmse"]
mae = test_metrics["mae"]
nasa = test_metrics["nasa_score"]

n_high = int((test_pred["risk_level"] == "HIGH").sum())
n_medium = int((test_pred["risk_level"] == "MEDIUM").sum())
n_low = int((test_pred["risk_level"] == "LOW").sum())
mean_predicted_rul = float(test_pred["predicted_RUL"].mean())

st.title("⚙️ NASA CMAPSS — Predictive Maintenance Dashboard")

st.markdown(
    """
Ce dashboard transforme les prédictions de **durée de vie résiduelle (RUL)** en outil d’aide à la décision pour la maintenance.

**Question métier principale : quels moteurs doivent être surveillés ou maintenus en priorité ?**

Le modèle prédit le nombre de cycles restants avant panne à partir des signaux capteurs du dataset **NASA CMAPSS FD001**.
"""
)

st.divider()

st.header("1. Vue opérationnelle — Priorisation de maintenance")

st.markdown(
    """
Les moteurs sont classés selon leur **RUL prédit**.  
Plus le RUL prédit est faible, plus la maintenance est prioritaire.
"""
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Moteurs critiques", n_high)
col2.metric("Moteurs moyens", n_medium)
col3.metric("Moteurs faibles", n_low)
col4.metric("RUL moyen prédit", f"{mean_predicted_rul:.1f} cycles")

st.markdown(
    """
**Règle de décision utilisée :**
- **Critique** : RUL prédit ≤ 30 cycles
- **Moyen** : RUL prédit ≤ 60 cycles
- **Faible** : RUL prédit > 60 cycles
"""
)

st.subheader("Moteurs à prioriser")

df_display = prepare_display_table(test_pred)
st.dataframe(df_display, use_container_width=True, hide_index=True)

st.divider()

st.header("2. Analyse détaillée d’un moteur")

st.markdown(
    """
Sélectionnez un moteur pour visualiser son niveau de risque, son RUL prédit et l’évolution de ses capteurs.  
Cette section permet de passer d’une vue globale à une analyse moteur par moteur.
"""
)

selected_unit = st.selectbox(
    "Sélectionner un moteur",
    options=sorted(test_features["unit_id"].unique()),
    format_func=lambda x: f"Moteur {x}",
)

unit_data = test_features[test_features["unit_id"] == selected_unit].copy()
unit_pred = test_pred[test_pred["unit_id"] == selected_unit].iloc[0]

unit_risk_label = RISK_LABELS.get(unit_pred["risk_level"], unit_pred["risk_level"])

col_a, col_b, col_c, col_d = st.columns(4)

col_a.metric("RUL prédit", f"{unit_pred['predicted_RUL']:.0f} cycles")
col_b.metric("RUL réel", f"{unit_pred['true_RUL']:.0f} cycles")
col_c.metric("Erreur", f"{unit_pred['absolute_error']:.0f} cycles")
col_d.metric("Niveau de risque", unit_risk_label)

sensor_cols = [
    col
    for col in test_features.columns
    if col.startswith("sensor_")
    and "rollmean" not in col
    and "delta" not in col
]

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
        name="Valeur capteur",
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
            name="Moyenne glissante — 5 cycles",
            line=dict(color="#378ADD", width=2),
        )
    )

fig_sensor.update_layout(
    title=f"Évolution de {selected_sensor} — Moteur {selected_unit}",
    xaxis_title="Cycle",
    yaxis_title="Valeur normalisée",
    legend=dict(orientation="h"),
)

st.plotly_chart(fig_sensor, use_container_width=True)

st.caption(
    "La courbe grise montre le signal capteur brut. La courbe bleue lisse le signal sur 5 cycles pour rendre la tendance plus lisible."
)

st.divider()

st.header("3. Performance du modèle — Peut-on faire confiance aux prédictions ?")

st.markdown(
    """
Cette section évalue la qualité du modèle sur les moteurs test.  
La diagonale du graphique représente une prédiction parfaite : plus les points sont proches de cette ligne, meilleure est la prédiction.
"""
)

col_m1, col_m2, col_m3 = st.columns(3)

col_m1.metric("RMSE", f"{rmse:.1f} cycles")
col_m2.metric("MAE", f"{mae:.1f} cycles")
col_m3.metric("Score NASA", f"{nasa:.0f}")

col_left, col_right = st.columns(2)

with col_left:
    fig_scatter = px.scatter(
        test_pred,
        x="true_RUL",
        y="predicted_RUL",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        hover_data={
            "unit_id": True,
            "absolute_error": True,
            "risk_level": True,
        },
        title="RUL prédit vs RUL réel",
        labels={
            "true_RUL": "RUL réel",
            "predicted_RUL": "RUL prédit",
            "risk_level": "Risque",
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
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        title="Distribution des erreurs absolues",
        labels={
            "absolute_error": "Erreur absolue (cycles)",
            "risk_level": "Risque",
        },
    )

    st.plotly_chart(fig_error, use_container_width=True)

st.divider()

st.header("4. Comparaison avec une baseline")

st.markdown(
    """
Le modèle principal, **Random Forest**, est comparé à une baseline simple, **Ridge Regression**.  
L’objectif est de vérifier que le modèle retenu apporte un gain par rapport à une approche plus simple.
"""
)

comparison_df = build_model_comparison(metrics)
st.dataframe(comparison_df, use_container_width=True)

fig_comparison = px.bar(
    comparison_df,
    x="Évaluation",
    y="RMSE",
    color="Modèle",
    barmode="group",
    title="Comparaison RMSE — Ridge baseline vs Random Forest",
    labels={"RMSE": "RMSE (cycles)"},
)

st.plotly_chart(fig_comparison, use_container_width=True)

st.divider()

st.markdown(
    """
### Résumé

Ce dashboard répond à trois questions :

1. **Quels moteurs sont prioritaires pour la maintenance ?**  
2. **Quel est l’état détaillé d’un moteur donné ?**  
3. **Quelle est la fiabilité du modèle de prédiction RUL ?**
"""
)