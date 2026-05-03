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
Entrée  : signaux capteurs + features temporelles
Sortie  : RUL estimé en nombre de cycles
Modèle  : Ridge Regression baseline + Random Forest Regressor
```

---

## Dataset

Le projet utilise le sous-ensemble **FD001** du dataset NASA CMAPSS.

FD001 correspond à un cas simplifié :

- une seule condition opérationnelle ;
- un seul mode de défaillance ;
- 100 moteurs dans le train set ;
- 100 moteurs dans le test set ;
- des séries temporelles moteur par moteur ;
- 21 capteurs disponibles.

Le train set contient des moteurs observés jusqu'à la panne.  
Le test set contient des moteurs tronqués avant la panne, avec un fichier séparé donnant le RUL réel final.

---

## Architecture du projet

```text
nasa-cmapss-rul-prediction/
│
├── data/
│   ├── raw/                    # Données NASA CMAPSS
│   └── processed/              # Données nettoyées, features, prédictions
│
├── docs/
│   ├── readme.txt              # Documentation NASA du dataset
│   ├── Damage_Propagation_Modeling.pdf
│   └── screenshots/            # Captures du dashboard
│
├── notebooks/
│   └── 01_eda_fd001.ipynb      # Exploration et analyse des données
│
├── src/
│   ├── ingestion/
│   │   └── load.py             # Chargement des données et calcul du RUL train
│   ├── preprocessing/
│   │   └── clean.py            # Nettoyage, suppression capteurs, cap RUL
│   ├── features/
│   │   └── engineer.py         # Rolling mean, delta, features temporelles
│   └── models/
│       └── train.py            # Baseline, Random Forest, évaluation, prédictions
│
├── models/                     # Artefacts sauvegardés
├── dashboard/
│   └── app.py                  # Dashboard Streamlit
│
├── requirements.txt
└── README.md
```

---

## Pipeline du projet

```text
Données NASA CMAPSS FD001
        │
        ▼
load.py
        ├── chargement train/test/RUL
        └── calcul du RUL train : max_cycle - cycle
        │
        ▼
01_eda_fd001.ipynb
        ├── distribution du RUL
        ├── analyse des capteurs
        ├── détection des capteurs quasi-constants
        └── justification des choix de preprocessing
        │
        ▼
clean.py
        ├── suppression des capteurs quasi-constants
        └── cap du RUL à 150 cycles
        │
        ▼
engineer.py
        ├── rolling mean sur 5 cycles
        ├── delta cycle à cycle
        └── construction de 24 features pour FD001
        │
        ▼
train.py
        ├── split train/validation par moteur
        ├── Ridge baseline
        ├── Random Forest
        ├── évaluation RMSE · MAE · score NASA
        └── génération des prédictions test
        │
        ▼
app.py
        └── dashboard Streamlit de priorisation maintenance
```

---

## Approche méthodologique

### 1. Exploration des données

L'EDA sur FD001 montre que plusieurs capteurs sont quasi-constants et n'apportent pas de signal de dégradation exploitable.

Ces capteurs sont supprimés afin de :

- réduire le bruit ;
- simplifier le modèle ;
- concentrer l'apprentissage sur les signaux réellement informatifs.

Pour FD001, **8 capteurs** sont retenus après analyse de la variance.

---

### 2. Preprocessing

| Étape | Choix | Justification |
|---|---|---|
| Suppression capteurs | écart-type < 0.5 | Seuil défini à partir de l'EDA pour retirer les capteurs quasi-constants |
| Cap RUL | 150 cycles | Hypothèse cohérente avec la plage des RUL test FD001 |
| Split validation | par moteur | Évite le data leakage entre les cycles d'un même moteur |

Le cap du RUL à 150 cycles est une hypothèse de modélisation : au-delà d'un certain horizon, l'objectif n'est pas de prédire très précisément une durée de vie lointaine, mais d'améliorer l'apprentissage sur les phases proches de la dégradation.

---

### 3. Feature engineering

Pour chaque capteur retenu, deux types de features temporelles sont ajoutés :

- **Rolling mean sur 5 cycles** : lisse les fluctuations locales et rend la tendance plus lisible.
- **Delta cycle à cycle** : capture la variation instantanée du signal.

Pour FD001 : 8 capteurs bruts + 8 rolling means + 8 deltas = **24 features**

---

### 4. Validation par moteur

Le split train/validation est effectué **par moteur** et non par ligne.

Cette étape est essentielle : si les cycles d'un même moteur étaient répartis à la fois dans le train et dans la validation, le modèle pourrait apprendre des patterns propres à ce moteur, ce qui créerait du **data leakage**.

- 80 moteurs → entraînement
- 20 moteurs → validation

---

## Métriques d'évaluation

Le modèle est évalué à l'aide de métriques de régression classiques ainsi que d'une métrique spécifique à la maintenance prédictive.

### MAE — Mean Absolute Error

La MAE mesure l'écart absolu moyen entre le RUL prédit et le RUL réel.

Elle est facilement interprétable car elle est exprimée directement en nombre de cycles :

> En moyenne, le modèle se trompe de X cycles.

### RMSE — Root Mean Squared Error

La RMSE mesure l'erreur moyenne du modèle en donnant plus de poids aux grandes erreurs.

Cette métrique est utile dans un contexte de maintenance prédictive, car les grandes erreurs de prédiction peuvent avoir un impact opérationnel important.

### NASA Score / PHM08 Score

Le NASA Score, aussi appelé PHM08 Score, est une métrique spécifique à la prédiction du Remaining Useful Life.

Contrairement à la MAE ou à la RMSE, il applique une pénalité asymétrique : les surestimations du RUL sont davantage pénalisées que les sous-estimations.

Cette logique est particulièrement adaptée à la maintenance prédictive :

- sous-estimer le RUL conduit à une maintenance anticipée ;
- surestimer le RUL peut conduire à une panne non anticipée.

Le score est défini par la formule suivante :

$$
s = \sum_{i=1}^{n}
\begin{cases}
\exp\left(-\frac{d_i}{13}\right) - 1, & \text{si } d_i < 0 \\
\exp\left(\frac{d_i}{10}\right) - 1, & \text{si } d_i \geq 0
\end{cases}
$$

avec :

$$
d_i = \widehat{RUL}_i - RUL_i
$$

où :

- $\widehat{RUL}_i$ correspond au RUL prédit ;
- $RUL_i$ correspond au RUL réel ;
- $d_i < 0$ signifie que le modèle sous-estime le RUL ;
- $d_i \geq 0$ signifie que le modèle surestime le RUL.

Le NASA Score n'est pas borné et ne s'interprète pas comme un pourcentage ou comme une note sur 100. Plus le score est faible, meilleur est le modèle. Un score de 0 correspondrait à des prédictions parfaites.

En pratique, cette métrique permet d'évaluer non seulement la précision statistique du modèle, mais aussi la criticité métier des erreurs de prédiction.

---

## Modélisation

Deux modèles sont comparés :

- **Ridge Regression** : baseline simple, linéaire et interprétable.
- **Random Forest Regressor** : modèle non linéaire capable de capter des interactions entre capteurs.

### Résultats validation — tous les cycles

| Modèle | RMSE | MAE | Score NASA |
|---|---|---|---|
| Ridge baseline | 24.73 | 19.39 | 58,702.96 |
| Random Forest | **21.60** | **15.98** | 91,895.86 |

Le Random Forest améliore la RMSE et la MAE par rapport à la baseline Ridge.  
Son score NASA est cependant plus élevé, ce qui indique que certaines erreurs sont davantage pénalisées par la métrique asymétrique PHM'08.

---

### Résultats test NASA — dernier cycle observé

| Métrique | Valeur |
|---|---|
| RMSE | **23.16 cycles** |
| MAE | **16.72 cycles** |
| Score NASA/PHM'08 | **5,331.90** |

Le test set NASA est évalué sur les 100 moteurs test, à partir du dernier cycle observé pour chaque moteur.  
Contrairement au train set, les moteurs test sont tronqués avant la panne : le RUL réel n'est donc pas nécessairement proche de zéro.

---

> ⚠️ **Note sur l'évaluation "last cycle"**  
> Une évaluation "last cycle" sur le split validation interne donne des scores très faibles, car les moteurs du train set sont observés jusqu'à la panne — le dernier cycle correspond donc à un RUL proche de zéro.  
> Cette métrique n'est pas directement comparable au test set NASA, où les moteurs sont volontairement tronqués avant défaillance.  
> **La performance principale à retenir : validation all cycles RMSE 21.60 — test NASA RMSE 23.16.**

---

## Dashboard Streamlit

Le dashboard répond à trois questions opérationnelles :

1. **Quels moteurs sont prioritaires pour la maintenance ?**  
   Vue opérationnelle avec niveaux de risque et table de priorisation.

2. **Quel est l'état détaillé d'un moteur donné ?**  
   Analyse individuelle avec RUL prédit, RUL réel, erreur, niveau de risque et courbes capteurs.

3. **Peut-on faire confiance aux prédictions ?**  
   Métriques RMSE, MAE, score NASA, comparaison baseline et visualisation des erreurs.

### Niveaux de risque utilisés

| Niveau | Règle | Action métier |
|---|---|---|
| Critique | RUL prédit ≤ 30 cycles | Maintenance prioritaire |
| À surveiller | RUL prédit ≤ 60 cycles | Maintenance à planifier |
| Stable | RUL prédit > 60 cycles | Surveillance normale |

---

## Ce que j'ai appris — Décisions techniques clés

- **Séparer preprocessing et feature engineering** permet de mieux contrôler ce qu'on donne au modèle et facilite les itérations.
- **Valider par moteur et non par ligne** est essentiel sur des données de séries temporelles pour éviter le data leakage.
- **Construire un score expliquable** avant d'appliquer du ML force à comprendre le domaine et valide les choix de features.
- **La métrique "last cycle" sur le train set est trompeuse** — les moteurs train vont jusqu'à la panne, contrairement au test set NASA.
- **Un Random Forest sans optimisation avancée** donne déjà des résultats solides sur CMAPSS FD001 et constitue une bonne baseline ML non linéaire.

---

## Stack technique

| Catégorie | Outils |
|---|---|
| Langage | Python |
| Données | pandas, numpy |
| Machine Learning | scikit-learn |
| Modèles | Ridge Regression, Random Forest |
| Évaluation | RMSE, MAE, score NASA/PHM'08 |
| Visualisation | Plotly, Streamlit |
| Notebook | Jupyter, matplotlib, seaborn |
| Sérialisation | joblib |

---

## Lancer le projet en local

```bash
# 1. Cloner le repository
git clone https://github.com/yhakam/nasa-cmapss-rul-prediction
cd nasa-cmapss-rul-prediction

# 2. Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\Activate.ps1  # Windows PowerShell

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Placer les fichiers NASA CMAPSS dans data/raw/
# Télécharger ici : https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

# 5. Lancer la pipeline
python src/ingestion/load.py
python src/preprocessing/clean.py
python src/features/engineer.py
python src/models/train.py

# 6. Lancer le dashboard
streamlit run dashboard/app.py
```

---

## Fichiers générés

Après exécution complète du pipeline :

```
data/processed/train_processed.csv
data/processed/test_processed.csv
data/processed/train_clean.csv
data/processed/test_clean.csv
data/processed/train_features.csv
data/processed/test_features.csv
data/processed/validation_predictions.csv
data/processed/test_predictions.csv
models/metrics.json
models/random_forest_rul.joblib
```

---

## Limites

- Pipeline calibrée sur **FD001** uniquement — la généralisation à FD002/FD003/FD004 nécessite une adaptation du preprocessing pour les conditions opérationnelles multiples.
- Le seuil de suppression des capteurs (σ < 0.5) et la fenêtre rolling mean (5 cycles) sont justifiés par l'EDA mais pourraient être optimisés.
- Le Random Forest ne modélise pas explicitement les dépendances temporelles longues.
- Pas d'optimisation des hyperparamètres — première approche ML classique.

---

## Améliorations futures

- Généraliser le pipeline aux datasets FD002, FD003 et FD004.
- Optimiser le cap RUL, la fenêtre rolling mean et les hyperparamètres par validation croisée.
- Tester des modèles dédiés aux séries temporelles : LSTM, GRU, Transformer.
- Ajouter l'importance des variables dans le dashboard.
- MLflow pour le suivi des expériences.
- Dockeriser l'application et exposer le modèle via FastAPI.

---

## Référence

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).  
*Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.*  
PHM'08, Denver, CO. — [PDF](docs/Damage_Propagation_Modeling.pdf)

*Données : [NASA CMAPSS Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) — open data, usage non commercial*