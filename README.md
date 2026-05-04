# NASA CMAPSS — Maintenance Prédictive & Prédiction de RUL

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![SQLite](https://img.shields.io/badge/SQLite-Analytics-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

Projet Data Science de bout en bout sur le dataset de référence **NASA CMAPSS FD001**.

L’objectif est de prédire la **durée de vie résiduelle** d’un moteur industriel, appelée **RUL — Remaining Useful Life**, à partir de signaux capteurs.  
Le projet couvre toute la chaîne : ingestion, exploration, preprocessing, feature engineering, modélisation, évaluation, couche analytique SQL et dashboard Streamlit orienté aide à la décision.

> *"The task was to estimate remaining life of an unspecified system using historical data only."*  
> — Saxena et al. (2008), PHM’08

---

## Résumé exécutif

Ce projet répond à une problématique industrielle simple :

> **Quels moteurs doivent être maintenus en priorité avant qu’une panne ne survienne ?**

À partir de données capteurs simulant le vieillissement de moteurs, un modèle de Machine Learning prédit le nombre de cycles restants avant panne.  
Les prédictions sont ensuite transformées en niveaux de risque opérationnels :

- **Critique** : maintenance prioritaire ;
- **À surveiller** : maintenance à planifier ;
- **Stable** : surveillance normale.

Le dashboard permet donc de passer d’une prédiction ML à une décision métier directement exploitable.

---

## Résultats clés

| Élément | Résultat |
|---|---:|
| Dataset | NASA CMAPSS FD001 |
| Nombre de moteurs test | 100 |
| Modèle retenu | Random Forest Regressor |
| RMSE test | **23.16 cycles** |
| MAE test | **16.72 cycles** |
| NASA Score test | **5,331.90** |
| Dashboard | Streamlit déployé en ligne |

Le modèle obtient une erreur absolue moyenne de **16.72 cycles** sur le test set NASA.  
Les prédictions sont ensuite converties en niveaux de risque afin de prioriser les moteurs nécessitant une intervention de maintenance.

---

## Dashboard

**Dashboard live** → [nasa-cmapss-rul-prediction.streamlit.app](https://nasa-cmapss-rul-prediction.streamlit.app/)

### Vue opérationnelle

La vue opérationnelle synthétise les moteurs critiques, les moteurs à surveiller et les moteurs stables afin de prioriser les actions de maintenance.

![Vue opérationnelle](assets/dashboard_overview.png)

### Table de priorisation

Les moteurs sont triés par niveau de risque puis par RUL prédit croissant. Les moteurs les plus urgents apparaissent en haut de la table.

![Table de priorisation](assets/priority_table.png)

### Analyse détaillée d’un moteur

Cette vue permet d’analyser un moteur spécifique : RUL prédit, RUL réel, erreur de prédiction, niveau de risque et évolution d’un capteur au fil des cycles.

![Analyse moteur](assets/engine_analysis.png)

---

## Contexte métier

La maintenance prédictive est un enjeu critique dans les secteurs industriels : énergie, aéronautique, manufacturing, transport ou Oil & Gas.

Une panne non anticipée peut entraîner :

- des arrêts de production coûteux ;
- des interventions de maintenance urgentes ;
- des risques de sécurité ;
- une mauvaise allocation des ressources techniques.

L’objectif de ce projet est donc d’estimer la durée de vie restante d’un moteur afin d’aider les équipes opérationnelles à prioriser les interventions.

---

## Objectif Data Science

À partir de l’historique des capteurs d’un moteur, le modèle doit estimer son **RUL**, c’est-à-dire le nombre de cycles restants avant panne.

Le problème est formulé comme une tâche de **régression supervisée** :

```text
Entrée  : signaux capteurs + features temporelles
Sortie  : RUL estimé en nombre de cycles
Modèles : Ridge Regression baseline + Random Forest Regressor
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

Le train set contient des moteurs observés jusqu’à la panne.  
Le test set contient des moteurs tronqués avant panne, avec un fichier séparé donnant le RUL réel final.

La documentation du dataset est conservée dans :

```text
docs/readme.txt
docs/Damage_Propagation_Modeling.pdf
```

Le PDF de Saxena et al. (2008) décrit notamment la génération des données, la simulation run-to-failure et la métrique utilisée dans la compétition PHM’08.

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
│   ├── raw/                        # Données NASA CMAPSS
│   └── processed/                  # Données nettoyées, features, prédictions
│
├── database/
│   └── cmapss.db                   # Base SQLite générée localement, non versionnée
│
├── docs/
│   ├── readme.txt                  # Documentation NASA du dataset
│   └── Damage_Propagation_Modeling.pdf
│
├── models/                         # Artefacts sauvegardés
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
│   │   └── load.py                 # Chargement des données et calcul du RUL train
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
db.py
├── création de la base SQLite
├── chargement des prédictions
├── chargement des métriques
└── chargement des données capteurs
│
▼
app.py
└── dashboard Streamlit — interroge SQLite si disponible
```

---

## Approche méthodologique

### 1. Exploration des données

L’EDA sur FD001 montre que plusieurs capteurs sont quasi constants et n’apportent pas de signal de dégradation exploitable.  
Ces capteurs sont supprimés afin de :

- réduire le bruit ;
- simplifier le modèle ;
- concentrer l’apprentissage sur les signaux réellement informatifs.

Pour FD001, **8 capteurs** sont retenus après analyse de la variance.

### 2. Preprocessing

| Étape | Choix | Justification |
|---|---|---|
| Suppression capteurs | écart-type < 0.5 | Seuil défini à partir de l’EDA pour retirer les capteurs quasi constants |
| Cap RUL | 150 cycles | Hypothèse cohérente avec la plage des RUL test FD001 |
| Split validation | par moteur | Évite le data leakage entre les cycles d’un même moteur |

Le cap du RUL à 150 cycles est une hypothèse de modélisation : au-delà d’un certain horizon, l’objectif n’est pas de prédire très précisément une durée de vie lointaine, mais d’améliorer l’apprentissage sur les phases proches de la dégradation.

### 3. Feature engineering

Pour chaque capteur retenu, deux types de features temporelles sont ajoutés :

- **Rolling mean sur 5 cycles** : lisse les fluctuations locales et rend la tendance plus lisible.
- **Delta cycle à cycle** : capture la variation instantanée du signal.

Pour FD001 :

```text
8 capteurs bruts + 8 rolling means + 8 deltas = 24 features
```

### 4. Validation par moteur

Le split train/validation est effectué **par moteur** et non par ligne.  
Cette étape est essentielle : si les cycles d’un même moteur étaient répartis à la fois dans le train et dans la validation, le modèle pourrait apprendre des patterns propres à ce moteur, ce qui créerait du **data leakage**.

```text
80 moteurs → entraînement
20 moteurs → validation
```

---

## SQL Analytics Layer

Les sorties du pipeline sont persistées dans une base **SQLite** (`database/cmapss.db`), générée localement avec :

```bash
python src/database/db.py
```

La base contient trois tables principales :

| Table | Contenu |
|---|---|
| `predictions` | RUL prédit, RUL réel, erreur absolue, niveau de risque par moteur |
| `metrics` | RMSE, MAE, score NASA pour chaque modèle et évaluation |
| `sensor_readings` | Signaux capteurs bruts et features temporelles par cycle |

Le dashboard interroge `predictions` via SQL pour trier et filtrer les moteurs selon leur niveau de risque.  
Le fichier `sql/queries.sql` contient des requêtes métier réutilisables :

- moteurs critiques ;
- répartition des niveaux de risque ;
- analyse des erreurs de prédiction ;
- RUL moyen par catégorie ;
- évolution de capteurs pour un moteur donné.

Pour exécuter les requêtes :

```bash
python src/database/run_queries.py
```

> La base SQLite n’est pas versionnée : elle est générée localement à partir des fichiers du pipeline.  
> Le dashboard bascule automatiquement sur les fichiers CSV si la base SQLite n’est pas disponible, par exemple sur Streamlit Cloud.

---

## Métriques d’évaluation

Le modèle est évalué à l’aide de métriques de régression classiques ainsi que d’une métrique spécifique à la maintenance prédictive.

L’objectif n’est pas seulement de mesurer l’erreur moyenne du modèle, mais aussi de comprendre la criticité métier des erreurs : dans un contexte industriel, surestimer la durée de vie restante d’un moteur peut être plus risqué que la sous-estimer.

### MAE — Mean Absolute Error

La MAE mesure l’écart absolu moyen entre le RUL prédit et le RUL réel.

```math
MAE = \frac{1}{n} \sum_{i=1}^{n} \left| RUL_i - \widehat{RUL}_i \right|
```

Cette métrique est facilement interprétable car elle est exprimée directement en nombre de cycles.

> Sur le test set NASA, le modèle se trompe en moyenne de **16.72 cycles**.

### RMSE — Root Mean Squared Error

```math
RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( RUL_i - \widehat{RUL}_i \right)^2 }
```

Elle mesure l’erreur moyenne du modèle en donnant plus de poids aux grandes erreurs.

> Sur le test set NASA, le modèle obtient une **RMSE de 23.16 cycles**.

### NASA Score / PHM’08 Score

Le NASA Score est une métrique asymétrique spécifique à la prédiction du RUL.  
Les surestimations du RUL sont davantage pénalisées que les sous-estimations, car elles peuvent retarder une intervention de maintenance.

```math
s = \sum_{i=1}^{n}
\begin{cases}
\exp\left(-\frac{d_i}{13}\right) - 1, & \text{si } d_i < 0 \\
\exp\left(\frac{d_i}{10}\right) - 1, & \text{si } d_i \geq 0
\end{cases}
\quad \text{avec} \quad
d_i = \widehat{RUL}_i - RUL_i
```

Le score n’est pas borné. **Plus il est faible, meilleur est le modèle.**

> Sur le test set NASA : **NASA Score de 5,331.90** sur 100 moteurs, soit environ **53.3 points par moteur** en moyenne.  
> Référence : `docs/Damage_Propagation_Modeling.pdf`, Saxena et al. (2008), Section VII.

---

## Modélisation

### Résultats validation — tous les cycles

| Modèle | RMSE | MAE | Score NASA |
|---|---:|---:|---:|
| Ridge baseline | 24.73 | 19.39 | **58,702.96** |
| Random Forest | **21.60** | **15.98** | 91,895.86 |

Le Random Forest améliore la RMSE et la MAE par rapport à la baseline Ridge.  
Son score NASA est cependant plus élevé, ce qui indique que certaines erreurs sont davantage pénalisées par la métrique asymétrique PHM’08.

Cette différence rappelle qu’un modèle peut réduire l’erreur moyenne tout en produisant quelques erreurs critiques plus coûteuses selon la métrique métier.

### Résultats test NASA — dernier cycle observé

| Métrique | Valeur |
|---|---:|
| RMSE | **23.16 cycles** |
| MAE | **16.72 cycles** |
| Score NASA / PHM’08 | **5,331.90** |

> **Note sur l’évaluation “last cycle”**  
> Une évaluation “last cycle” sur le split validation interne peut donner des scores artificiellement faibles, car les moteurs du train set sont observés jusqu’à la panne : leur dernier cycle correspond donc à un RUL proche de zéro.  
> Cette métrique n’est pas directement comparable au test set NASA, où les moteurs sont volontairement tronqués avant défaillance.  
> La performance principale à retenir est donc : **validation all cycles RMSE 21.60 — test NASA RMSE 23.16**.

---

## Interprétation des résultats

Le Random Forest obtient de meilleures performances que la baseline Ridge sur les métriques classiques, avec une RMSE validation de **21.60 cycles** et une RMSE test de **23.16 cycles**.

Cependant, son score NASA validation est supérieur à celui de la baseline Ridge. Cela signifie que, même si le Random Forest réduit l’erreur moyenne, certaines erreurs sont plus fortement pénalisées par la métrique asymétrique PHM’08, notamment les surestimations du RUL.

Le Random Forest est retenu comme modèle principal car il offre le meilleur compromis global sur les métriques classiques.  
Dans un contexte industriel réel, le choix final du modèle devrait aussi intégrer :

- la fréquence des surestimations ;
- le coût d’une maintenance anticipée ;
- le coût d’une panne non anticipée ;
- la criticité des équipements concernés.

---

## Dashboard Streamlit

Le dashboard répond à trois questions opérationnelles :

1. **Quels moteurs sont prioritaires pour la maintenance ?**
2. **Quel est l’état détaillé d’un moteur donné ?**
3. **Peut-on faire confiance aux prédictions ?**

### Niveaux de risque

| Niveau | Règle | Action métier |
|---|---|---|
| Critique | RUL prédit ≤ 30 cycles | Maintenance prioritaire |
| À surveiller | RUL prédit ≤ 60 cycles | Maintenance à planifier |
| Stable | RUL prédit > 60 cycles | Surveillance normale |

---

## Ce que j’ai appris — décisions techniques clés

- **Séparer preprocessing et feature engineering** permet de mieux contrôler ce qui est donné au modèle et facilite les itérations.
- **Valider par moteur et non par ligne** est essentiel sur des données de séries temporelles pour éviter le data leakage.
- **Persister les prédictions en base SQLite** rapproche le projet d’un workflow analytique en entreprise et permet d’écrire des requêtes métier directement exploitables.
- **La métrique “last cycle” sur le train set est trompeuse** : les moteurs train vont jusqu’à la panne, contrairement au test set NASA.
- **Un Random Forest sans optimisation avancée** constitue une baseline non linéaire solide, mais son score NASA montre qu’une amélioration future devrait cibler les surestimations du RUL.

---

## Stack technique

| Catégorie | Outils |
|---|---|
| Langage | Python |
| Données | pandas, numpy |
| Machine Learning | scikit-learn |
| Modèles | Ridge Regression, Random Forest |
| Base de données | SQLite |
| Évaluation | RMSE, MAE, score NASA / PHM’08 |
| Visualisation | Plotly, Streamlit |
| Notebook | Jupyter, matplotlib |
| Sérialisation | joblib |

---

## Lancer le projet en local

### 1. Cloner le repository

```bash
git clone https://github.com/yhakam/nasa-cmapss-rul-prediction
cd nasa-cmapss-rul-prediction
```

### 2. Créer et activer l’environnement virtuel

```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Ajouter les données NASA CMAPSS

Télécharger le dataset **CMAPSS Jet Engine Simulated Data** depuis le portail NASA Open Data, puis placer les fichiers bruts dans :

```text
data/raw/
```

Les fichiers attendus pour FD001 sont notamment :

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

La documentation fournie avec le dataset est conservée dans :

```text
docs/readme.txt
docs/Damage_Propagation_Modeling.pdf
```

Le PDF de Saxena et al. (2008) décrit la génération des données, le contexte de simulation et la métrique utilisée dans la compétition PHM’08.

### 5. Lancer le pipeline complet

```bash
python src/ingestion/load.py
python src/preprocessing/clean.py
python src/features/engineer.py
python src/models/train.py
```

Ces scripts génèrent les fichiers intermédiaires dans `data/processed/` ainsi que les artefacts du modèle dans `models/`.

### 6. Construire la base SQLite

```bash
python src/database/db.py
```

Cette commande crée localement :

```text
database/cmapss.db
```

La base contient les prédictions, les métriques et les données capteurs utilisées pour les analyses SQL.

### 7. Exécuter les requêtes métier SQL

```bash
python src/database/run_queries.py
```

Les requêtes sont définies dans :

```text
sql/queries.sql
```

Elles permettent notamment d’identifier les moteurs critiques, d’analyser les erreurs du modèle et de suivre les niveaux de risque.

### 8. Lancer le dashboard Streamlit

```bash
streamlit run dashboard/app.py
```

Le dashboard utilise SQLite si la base est disponible.  
Si la base n’existe pas, il bascule automatiquement sur les fichiers CSV générés par le pipeline.

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
database/cmapss.db   # générée par db.py, non versionnée
```

---

## Limites

- Pipeline calibré sur **FD001** uniquement : la généralisation à FD002, FD003 et FD004 nécessite une adaptation du preprocessing.
- Le seuil de suppression des capteurs (`std < 0.5`) et la fenêtre rolling mean de 5 cycles sont justifiés par l’EDA, mais pourraient être optimisés.
- Le Random Forest ne modélise pas explicitement les dépendances temporelles longues.
- Pas d’optimisation avancée des hyperparamètres : le projet privilégie une première approche claire, robuste et interprétable.

---

## Améliorations futures

- Généraliser le pipeline aux datasets FD002, FD003 et FD004.
- Optimiser le cap RUL, la fenêtre rolling mean et les hyperparamètres par validation croisée.
- Analyser séparément les erreurs de sous-estimation et de surestimation du RUL.
- Tester des modèles dédiés aux séries temporelles : LSTM, GRU, Transformer.
- Ajouter l’importance des variables dans le dashboard.
- Ajouter MLflow pour le suivi des expériences.
- Dockeriser l’application et exposer le modèle via FastAPI.

---

## Références

Ce projet s’appuie sur le dataset **NASA CMAPSS FD001**, largement utilisé comme référence pour les travaux de maintenance prédictive et de prédiction de durée de vie résiduelle.

### Article de référence

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).  
*Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.*  
International Conference on Prognostics and Health Management, PHM’08, Denver, CO.

Le document est inclus dans le repository :

```text
docs/Damage_Propagation_Modeling.pdf
```

Il décrit notamment :

- le contexte de simulation des moteurs ;
- la génération des trajectoires run-to-failure ;
- la structure des données CMAPSS ;
- la tâche de prédiction de RUL ;
- le score asymétrique utilisé dans la compétition PHM’08.

### Documentation du dataset

La documentation NASA du dataset est également conservée dans :

```text
docs/readme.txt
```

Elle précise la structure des fichiers, les colonnes disponibles, les conditions opérationnelles et les capteurs.

### Données

Dataset : [NASA CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

Les données sont utilisées dans un cadre pédagogique et non commercial.
