# NASA CMAPSS — Maintenance Prédictive & Prédiction de RUL

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-green)

Projet Data Science de bout en bout sur le dataset de référence **NASA CMAPSS FD001**.

L'objectif est de prédire la **durée de vie résiduelle** d'un moteur industriel, appelée **RUL — Remaining Useful Life**, à partir de signaux capteurs.  
Le projet couvre toute la chaîne : ingestion, exploration, preprocessing, feature engineering, modélisation, évaluation et dashboard Streamlit orienté aide à la décision.

> *"The task was to estimate remaining life of an unspecified system using historical data only."*  
> — Saxena et al. (2008), PHM'08

---

## Résumé exécutif

Ce projet répond à une problématique industrielle simple :

> **Quels moteurs doivent être maintenus en priorité avant qu'une panne ne survienne ?**

À partir de données capteurs simulant le vieillissement de moteurs, un modèle de Machine Learning prédit le nombre de cycles restants avant panne.  
Les prédictions sont ensuite transformées en niveaux de risque opérationnels :

- **Critique** : maintenance prioritaire
- **À surveiller** : maintenance à planifier
- **Stable** : surveillance normale

Le dashboard permet donc de passer d'une prédiction ML à une décision métier directement exploitable.

---

## Dashboard

**Dashboard live** → [nasa-cmapss-rul-prediction.streamlit.app](https://nasa-cmapss-rul-prediction.streamlit.app/)

### Vue opérationnelle

La vue opérationnelle synthétise les moteurs critiques, les moteurs à surveiller et les moteurs stables afin de prioriser les actions de maintenance.

![Vue opérationnelle](assets/dashboard_overview.png)

### Table de priorisation

Les moteurs sont triés par niveau de risque puis par RUL prédit croissant. Les moteurs les plus urgents apparaissent en haut de la table.

![Table de priorisation](assets/priority_table.png)

### Analyse détaillée d'un moteur

Cette vue permet d'analyser un moteur spécifique : RUL prédit, RUL réel, erreur de prédiction, niveau de risque et évolution d'un capteur au fil des cycles.

![Analyse moteur](assets/engine_analysis.png)

---

## Contexte métier

La maintenance prédictive est un enjeu critique dans les secteurs industriels : énergie, aéronautique, manufacturing, transport ou Oil & Gas.

Une panne non anticipée peut entraîner :

- des arrêts de production coûteux ;
- des interventions de maintenance urgentes ;
- des risques de sécurité ;
- une mauvaise allocation des ressources techniques.

L'objectif de ce projet est donc de prédire le plus tôt possible la durée de vie restante d'un moteur, afin d'aider les équipes opérationnelles à prioriser les interventions.

---

## Objectif Data Science

À partir de l'historique des capteurs d'un moteur, le modèle doit estimer son **RUL**, c'est-à-dire le nombre de cycles restants avant panne.

Le problème est formulé comme une tâche de **régression supervisée** :

```text