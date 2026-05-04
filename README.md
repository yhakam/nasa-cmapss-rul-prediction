# NASA CMAPSS — Maintenance Prédictive & Prédiction de RUL

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![SQLite](https://img.shields.io/badge/SQLite-Analytics-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

Projet Data Science de bout en bout sur le dataset de référence **NASA CMAPSS FD001**.

L'objectif est de prédire la **durée de vie résiduelle** d'un moteur industriel — appelée **RUL (Remaining Useful Life)** — à partir de signaux capteurs, puis de convertir ces prédictions en décisions de maintenance opérationnelles.

Le projet couvre la chaîne complète : ingestion, exploration, preprocessing, feature engineering, modélisation, couche analytique SQL et dashboard Streamlit orienté aide à la décision.

> *"The task was to estimate remaining life of an unspecified system using historical data only."*
> — Saxena et al. (2008), PHM'08 — [PDF](docs/Damage_Propagation_Modeling.pdf)

---

## Résumé exécutif

Ce projet répond à une problématique industrielle directe :

> **Quels moteurs doivent être maintenus en priorité avant qu'une panne ne survienne ?**

À partir de données capteurs simulant le vieillissement de moteurs, un modèle de Machine Learning prédit le nombre de cycles restants avant panne. Les prédictions sont ensuite converties en niveaux de risque opérationnels :

- **Critique** : maintenance prioritaire
- **A surveiller** : maintenance à planifier
- **Stable** : surveillance normale

Le dashboard permet de passer d'une prédiction ML à une décision métier directement exploitable.

---

## Résultats clés

| Élément | Résultat |
|---|---:|
| Dataset | NASA CMAPSS FD001 |
| Nombre de moteurs test | 100 |
| Modèle retenu | Random Forest Regressor |
| RMSE test | **23.16 cycles** |
| MAE test | **16.72 cycles** |
| NASA Score test | **5 331.90** |
| Dashboard | Streamlit déployé en ligne |

Le modèle se trompe en moyenne de **16.72 cycles** sur le test set NASA. Les prédictions sont ensuite transformées en niveaux de risque afin de prioriser les moteurs nécessitant une intervention.

---

## Dashboard

**Dashboard live** : [nasa-cmapss-rul-prediction.streamlit.app](https://nasa-cmapss-rul-prediction.streamlit.app/)

### Vue opérationnelle

Synthèse des moteurs critiques, à surveiller et stables pour prioriser les actions de maintenance.

![Vue opérationnelle](assets/dashboard_overview.png)

### Table de priorisation

Les moteurs sont triés par niveau de risque puis par RUL prédit croissant. Les cas les plus urgents apparaissent en haut de la table.

![Table de priorisation](assets/priority_table.png)

### Analyse détaillée d'un moteur

RUL prédit, RUL réel, erreur de prédiction, niveau de risque et évolution d'un capteur au fil des cycles.

![Analyse moteur](assets/engine_analysis.png)

---

## Contexte métier

La maintenance prédictive est un enjeu critique dans les secteurs industriels : énergie, aéronautique, manufacturing, transport, Oil & Gas.

Une panne non anticipée peut entraîner :

- des arrêts de production coûteux
- des interventions de maintenance urgentes et non planifiées
- des risques de sécurité pour les opérateurs
- une mauvaise allocation des ressources techniques

L'objectif est d'estimer la durée de vie restante d'un moteur afin d'aider les équipes opérationnelles à prioriser leurs interventions avant toute défaillance.

---

## Objectif Data Science

À partir de l'historique des capteurs d'un moteur, le modèle estime son **RUL** — le nombre de cycles restants avant panne. Le problème est formulé comme une tâche de **régression supervisée** :

```text
Entree  : signaux capteurs + features temporelles
Sortie  : RUL estimé en nombre de cycles
Modeles : Ridge Regression (baseline) + Random Forest Regressor
```

---

## Dataset

Le projet utilise le sous-ensemble **FD001** du dataset NASA CMAPSS (Saxena et al., 2008).

FD001 correspond à un cas simplifié :

- une seule condition opérationnelle
- un seul mode de défaillance
- 100 moteurs en train set, 100 moteurs en test set
- des séries temporelles moteur par moteur
- 21 capteurs disponibles

Le train set contient des moteurs observés jusqu'à la panne. Le test set contient des moteurs tronqués avant panne, avec un fichier séparé donnant le RUL réel final.

Source : [NASA CMAPSS Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) — open data, usage non commercial.
Documentation originale NASA : [readme.txt](docs/readme.txt)
Article de référence : [Damage Propagation Modeling (PDF)](docs/Damage_Propagation_Modeling.pdf)

---

## Architecture du projet

```text
nasa-cmapss-rul-prediction/
│
├── assets/                         # Captures du dashboard
│
├── dashboard/
│   └── app.py                      # Dashboard Streamlit
│
├── data/
│   ├── raw/                        # Données NASA CMAPSS brutes
│   └── processed/                  # Données nettoyées, features, prédictions
│
├── database/
│   └── cmapss.db                   # Base SQLite (generée localement, non versionnée)
│
├── docs/
│   ├── readme.txt                  # Documentation NASA originale du dataset
│   └── Damage_Propagation_Modeling.pdf  # Article PHM'08 — Saxena et al. (2008)
│
├── models/                         # Artefacts modèles sauvegardés
│
├── notebooks/
│   └── 01_eda_fd001.ipynb          # Exploration et analyse des données
│
├── sql/
│   └── queries.sql                 # Requêtes métier réutilisables
│
├── src/
│   ├── database/
│   │   ├── db.py                   # Construction de la base SQLite
│   │   └── run_queries.py          # Exécution des requêtes métier
│   ├── features/
│   │   └── engineer.py             # Rolling mean, delta, features temporelles
│   ├── ingestion/
│   │   └── load.py                 # Chargement des données + calcul du RUL train
│   ├── models/
│   │   └── train.py                # Baseline, Random Forest, évaluation, prédictions
│   └── preprocessing/
│       └── clean.py                # Nettoyage, suppression capteurs, cap RUL
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
├── chargement train / test / RUL réels
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
├── suppression des capteurs quasi-constants (sigma < 0.5)
└── cap du RUL à 150 cycles
│
▼
engineer.py
├── rolling mean sur 5 cycles par capteur
├── delta cycle à cycle par capteur
└── construction de 24 features pour FD001
│
▼
train.py
├── split train / validation par moteur (80/20)
├── Ridge Regression (baseline)
├── Random Forest Regressor
├── évaluation RMSE · MAE · score NASA PHM'08
└── génération des prédictions test
│
▼
db.py
├── création de la base SQLite
├── chargement des prédictions, métriques et capteurs
└── couche analytique SQL réutilisable
│
▼
app.py
└── dashboard Streamlit — interroge SQLite si disponible, CSV sinon
```

---

## Approche méthodologique

### 1. Exploration des données (EDA)

L'EDA sur FD001 montre que plusieurs capteurs sont quasi constants et n'apportent aucun signal de dégradation exploitable. Ces capteurs sont supprimés afin de réduire le bruit, simplifier le modèle et concentrer l'apprentissage sur les signaux réellement informatifs.

Pour FD001, **8 capteurs** sont retenus après analyse de la variance.

### 2. Preprocessing

| Étape | Choix | Justification |
|---|---|---|
| Suppression capteurs | écart-type < 0.5 | Seuil défini par l'EDA pour éliminer les capteurs quasi constants |
| Cap RUL | 150 cycles | Cohérent avec la plage des RUL test FD001 — améliore l'apprentissage sur les phases de dégradation proche |
| Split validation | par moteur | Evite le data leakage entre les cycles d'un même moteur |

### 3. Feature engineering

Pour chaque capteur retenu, deux features temporelles sont construites :

- **Rolling mean sur 5 cycles** : lisse les fluctuations locales et rend la tendance de dégradation plus lisible
- **Delta cycle à cycle** : capture la variation instantanée du signal

```text
8 capteurs bruts + 8 rolling means + 8 deltas = 24 features pour FD001
```

### 4. Validation par moteur

Le split train/validation est effectué par moteur et non par ligne. Si les cycles d'un même moteur étaient répartis dans les deux ensembles, le modèle pourrait apprendre des patterns propres à ce moteur — ce qui constituerait un **data leakage**.

```text
80 moteurs → entraînement
20 moteurs → validation
```

---

## SQL Analytics Layer

Les sorties du pipeline sont persistées dans une base **SQLite** (`database/cmapss.db`), générée localement :

```bash
python src/database/db.py
```

| Table | Contenu |
|---|---|
| `predictions` | RUL prédit, RUL réel, erreur absolue, niveau de risque par moteur |
| `metrics` | RMSE, MAE, score NASA pour chaque modèle |
| `sensor_readings` | Signaux capteurs bruts et features temporelles par cycle |

Le fichier `sql/queries.sql` contient des requêtes métier réutilisables :

- moteurs critiques
- répartition des niveaux de risque
- analyse des erreurs de prédiction
- RUL moyen par catégorie
- évolution d'un capteur pour un moteur donné

```bash
python src/database/run_queries.py
```

> La base SQLite n'est pas versionnée — elle est générée localement à partir des fichiers du pipeline. Le dashboard bascule automatiquement sur les fichiers CSV si la base n'est pas disponible (par exemple sur Streamlit Cloud).

---

## Métriques d'évaluation

Le modèle est évalué via des métriques classiques de régression et une métrique spécifique à la maintenance prédictive.

### MAE — Mean Absolute Error

```text
MAE = (1/n) * sum( |RUL_i - RUL_pred_i| )
```

Directement interprétable en nombre de cycles. Sur le test set NASA : **16.72 cycles d'erreur absolue moyenne**.

### RMSE — Root Mean Squared Error

```text
RMSE = sqrt( (1/n) * sum( (RUL_i - RUL_pred_i)^2 ) )
```

Donne plus de poids aux grandes erreurs. Sur le test set NASA : **23.16 cycles**.

### NASA Score / PHM'08 Score

Métrique asymétrique propre à la prédiction du RUL. Les surestimations sont davantage pénalisées que les sous-estimations, car elles peuvent retarder une intervention critique.

```text
d_i = RUL_pred_i - RUL_i

Si d_i < 0 : exp(-d_i / 13) - 1     (sous-estimation — pénalité faible)
Si d_i >= 0 : exp(d_i / 10) - 1      (surestimation — pénalité forte)

Score = sum sur tous les moteurs
```

Le score n'est pas borné. **Plus il est faible, meilleur est le modèle.**
Sur le test set NASA : **5 331.90** pour 100 moteurs, soit environ 53.3 points par moteur en moyenne.

---

## Modélisation

### Résultats validation — tous les cycles

| Modèle | RMSE | MAE | Score NASA |
|---|---:|---:|---:|
| Ridge baseline | 24.73 | 19.39 | **58 702.96** |
| Random Forest | **21.60** | **15.98** | 91 895.86 |

Le Random Forest améliore la RMSE et la MAE par rapport à la baseline Ridge. Son score NASA est cependant plus élevé, ce qui indique que certaines erreurs sont davantage pénalisées par la métrique asymétrique. Ce résultat rappelle qu'un modèle peut réduire l'erreur moyenne tout en produisant quelques surestimations coûteuses selon la métrique métier.

### Résultats test NASA — dernier cycle observé

| Métrique | Valeur |
|---|---:|
| RMSE | **23.16 cycles** |
| MAE | **16.72 cycles** |
| Score NASA / PHM'08 | **5 331.90** |

> **Note sur l'évaluation "last cycle"**
> Une évaluation "last cycle" sur le split validation interne peut donner des scores artificiellement faibles, car les moteurs du train set sont observés jusqu'à la panne : leur dernier cycle correspond à un RUL proche de zéro.
> Cette métrique n'est pas directement comparable au test set NASA, où les moteurs sont volontairement tronqués avant défaillance.
> La performance principale à retenir est donc : **validation all cycles RMSE 21.60 — test NASA RMSE 23.16**.

---

## Interprétation des résultats

Le Random Forest offre le meilleur compromis global sur les métriques classiques, avec une RMSE validation de **21.60 cycles** et une RMSE test de **23.16 cycles**.

Son score NASA validation est cependant supérieur à celui de la baseline Ridge. Même si le modèle réduit l'erreur moyenne, certaines surestimations du RUL sont plus fortement pénalisées par la métrique asymétrique PHM'08.

Le Random Forest est retenu comme modèle principal. Dans un contexte industriel réel, le choix final devrait aussi intégrer :

- la fréquence des surestimations
- le coût d'une maintenance anticipée
- le coût d'une panne non anticipée
- la criticité des équipements concernés

---

## Dashboard Streamlit

Le dashboard répond à trois questions opérationnelles :

1. **Quels moteurs sont prioritaires pour la maintenance ?**
2. **Quel est l'état détaillé d'un moteur donné ?**
3. **Peut-on faire confiance aux prédictions ?**

### Niveaux de risque

| Niveau | Règle | Action métier |
|---|---|---|
| Critique | RUL prédit <= 30 cycles | Maintenance prioritaire |
| A surveiller | RUL prédit <= 60 cycles | Maintenance à planifier |
| Stable | RUL prédit > 60 cycles | Surveillance normale |

---

## Ce que j'ai appris — décisions techniques clés

- **Séparer preprocessing et feature engineering** permet de contrôler précisément ce qui est donné au modèle et facilite les itérations.
- **Valider par moteur et non par ligne** est essentiel sur des données temporelles pour éviter le data leakage.
- **Persister les prédictions en SQLite** rapproche le projet d'un workflow analytique en entreprise et permet d'écrire des requêtes métier directement exploitables.
- **La métrique "last cycle" sur le train set est trompeuse** : les moteurs train vont jusqu'à la panne, contrairement au test set NASA — les deux ne sont pas comparables.
- **Un Random Forest sans optimisation avancée** constitue une baseline non linéaire solide, mais son score NASA montre qu'une amélioration future devrait cibler les surestimations du RUL.

---

## Stack technique

| Catégorie | Outils |
|---|---|
| Langage | Python 3.11 |
| Données | pandas, numpy |
| Machine Learning | scikit-learn |
| Modèles | Ridge Regression, Random Forest |
| Base de données | SQLite |
| Evaluation | RMSE, MAE, NASA Score / PHM'08 |
| Visualisation | Plotly, Streamlit |
| Notebook | Jupyter, matplotlib |
| Sérialisation | joblib |

---

## Lancer le projet en local

```bash
# 1. Cloner le repo
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

# 6. Construire la base SQLite (optionnel)
python src/database/db.py

# 7. Executer les requêtes SQL métier (optionnel)
python src/database/run_queries.py

# 8. Lancer le dashboard
streamlit run dashboard/app.py
```

---

## Fichiers générés

```text
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
database/cmapss.db              ← générée par db.py, non versionnée
```

---

## Limites

- Pipeline calibré sur **FD001** uniquement — la généralisation à FD002, FD003 et FD004 nécessite une adaptation du preprocessing pour les conditions opérationnelles multiples.
- Le seuil de suppression des capteurs (σ < 0.5) et la fenêtre rolling mean de 5 cycles sont justifiés par l'EDA mais pourraient être optimisés.
- Le Random Forest ne modélise pas explicitement les dépendances temporelles longues.
- Pas d'optimisation avancée des hyperparamètres — le projet privilégie une première approche claire, robuste et interprétable.

---

## Améliorations futures

- Généraliser le pipeline aux datasets FD002, FD003 et FD004.
- Optimiser le cap RUL, la fenêtre rolling mean et les hyperparamètres par validation croisée.
- Analyser séparément les erreurs de sous-estimation et de surestimation du RUL.
- Tester des modèles dédiés aux séries temporelles : LSTM, GRU, Transformer.
- Ajouter l'importance des variables dans le dashboard.
- MLflow pour le suivi des expériences.
- Dockeriser l'application et exposer le modèle via FastAPI.

---

## Référence

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
*Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.*
PHM'08, Denver, CO. — [PDF](docs/Damage_Propagation_Modeling.pdf)

Documentation NASA originale du dataset : [docs/readme.txt](docs/readme.txt)

Données : [NASA CMAPSS Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) — open data, usage non commercial.
