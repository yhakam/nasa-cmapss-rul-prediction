# NASA CMAPSS — Maintenance Prédictive & Prédiction de RUL

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red)
![License](https://img.shields.io/badge/License-MIT-green)

Projet de maintenance prédictive de bout en bout sur le dataset de référence **NASA CMAPSS FD001**.

L'objectif est de prédire la **durée de vie résiduelle** d'un moteur industriel, appelée **RUL — Remaining Useful Life**, à partir de signaux capteurs.  
Le projet couvre toute la chaîne Data Science : ingestion, exploration, preprocessing, feature engineering, modélisation, évaluation et dashboard Streamlit orienté aide à la décision.

> *"The task was to estimate remaining life of an unspecified system using historical data only."*  
> — Saxena et al. (2008), PHM'08

---

## Dashboard

**Dashboard live** → [nasa-cmapss-rul-prediction.streamlit.app](https://nasa-cmapss-rul-prediction.streamlit.app/)

> Captures à ajouter : vue opérationnelle, analyse moteur, performance modèle.

---

## Contexte métier

La maintenance prédictive est un enjeu critique dans les secteurs industriels : énergie, aéronautique, manufacturing, Oil & Gas.  
Anticiper la défaillance d'un équipement permet de réduire les arrêts non planifiés, d'optimiser les coûts de maintenance et d'améliorer la sécurité opérationnelle.

Ce projet modélise ce problème à partir de données de turbomoteurs simulées par la NASA dans le cadre de la compétition **Prognostics and Health Management 2008 — PHM'08**.

---

## Objectif du projet

À partir de l'historique capteur d'un moteur, le modèle doit estimer le nombre de cycles restants avant panne.

Le dashboard transforme ensuite cette prédiction en décision opérationnelle :

- **Critique** : RUL prédit ≤ 30 cycles → maintenance prioritaire
- **À surveiller** : RUL prédit ≤ 60 cycles → maintenance à planifier
- **Stable** : RUL prédit > 60 cycles → surveillance normale

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
│   └── Damage_Propagation_Modeling.pdf
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
│       └── train.py            # Ridge baseline, Random Forest, évaluation
│
├── models/                     # Artefacts sauvegardés
├── dashboard/
│   └── app.py                  # Dashboard Streamlit
│
├── requirements.txt
└── README.md
```

---

## Pipeline

```text
Données NASA CMAPSS FD001
        │
        ▼
load.py
        ├── chargement train/test/RUL
        └── calcul du RUL train : max_cycle - cycle
        │
        ▼
clean.py
        ├── suppression des capteurs quasi-constants (σ < 0.5)
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
        ├── split train/validation par moteur (80/20)
        ├── Ridge baseline
        ├── Random Forest
        └── évaluation RMSE · MAE · score NASA
        │
        ▼
app.py
        └── dashboard Streamlit de priorisation maintenance
```

---

## Approche méthodologique

### 1. Exploration des données

L'EDA sur FD001 montre que plusieurs capteurs sont quasi-constants et n'apportent pas de signal de dégradation exploitable.  
Ces capteurs sont supprimés afin de réduire le bruit et de concentrer la modélisation sur les signaux réellement informatifs.  
Pour FD001, **8 capteurs** sont retenus après analyse de la variance.

> Saxena et al. (2008), p.5 — Section V.B (Noise) — [PDF](docs/Damage_Propagation_Modeling.pdf)

### 2. Preprocessing

| Étape | Choix | Justification |
|---|---|---|
| Suppression capteurs | écart-type < 0.5 | Seuil défini à partir de l'EDA pour retirer les capteurs quasi-constants |
| Cap RUL | 150 cycles | Cohérent avec la plage des RUL test FD001 (10-150 cycles) |
| Split validation | par moteur | Évite le data leakage entre les cycles d'un même moteur |

> Saxena et al. (2008), p.7 — Section VI — [PDF](docs/Damage_Propagation_Modeling.pdf)

### 3. Feature engineering

Pour chaque capteur retenu, deux types de features temporelles sont ajoutés :

- **Rolling mean sur 5 cycles** : lisse les fluctuations locales et rend la tendance plus lisible
- **Delta cycle à cycle** : capture la variation instantanée du signal

Pour FD001 : 8 capteurs bruts + 8 rolling means + 8 deltas = **24 features**

### 4. Validation par moteur

Le split train/validation est effectué **par moteur** et non par ligne.  
Si les cycles d'un même moteur étaient répartis entre train et validation, le modèle apprendrait des patterns propres à ce moteur — ce qui créerait du **data leakage**.

- 80 moteurs → entraînement
- 20 moteurs → validation

---

## Modélisation

Deux modèles sont comparés :

- **Ridge Regression** : baseline simple et interprétable
- **Random Forest Regressor** : modèle non linéaire capable de capter des interactions entre capteurs

### Résultats validation — tous les cycles

| Modèle | RMSE | MAE | Score NASA |
|---|---|---|---|
| Ridge baseline | 24.73 | 19.39 | 58,702.96 |
| Random Forest | **21.60** | **15.98** | 91,895.86 |

Le Random Forest améliore la RMSE et la MAE par rapport à la baseline Ridge.  
Son score NASA est cependant plus élevé, ce qui montre que certaines erreurs sont davantage pénalisées par la métrique asymétrique PHM'08.

### Résultats test NASA — dernier cycle observé

| Métrique | Valeur |
|---|---|
| RMSE | **23.16 cycles** |
| MAE | **16.72 cycles** |
| Score NASA/PHM'08 | **5,331.90** |

> ⚠️ **Note sur l'évaluation "last cycle"**  
> Une évaluation "last cycle" sur la validation interne donne des scores très faibles (RMSE 2.60), car les moteurs du train set sont observés jusqu'à la panne — le dernier cycle correspond donc à un RUL proche de zéro.  
> Cette métrique n'est pas directement comparable au test set NASA, où les moteurs sont volontairement tronqués avant défaillance.  
> **La performance principale à retenir est : validation all cycles RMSE 21.60 — test NASA RMSE 23.16.**

---

## Dashboard Streamlit

Le dashboard répond à trois questions opérationnelles :

1. **Quels moteurs sont prioritaires pour la maintenance ?**  
   Vue opérationnelle avec niveaux de risque et table de priorisation.

2. **Quel est l'état détaillé d'un moteur donné ?**  
   Analyse individuelle avec RUL prédit, RUL réel, erreur, niveau de risque et courbes capteurs.

3. **Peut-on faire confiance aux prédictions ?**  
   Métriques RMSE, MAE, score NASA, comparaison baseline et visualisation des erreurs.

---

## Stack technique

| Catégorie | Outils |
|---|---|
| Langage | Python 3.11 |
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

# 6. Lancer le dashboard
streamlit run dashboard/app.py
```

---

## Fichiers générés

Après exécution complète de la pipeline :

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

- Pipeline calibrée sur **FD001** uniquement — la généralisation à FD002/FD003/FD004 nécessite une adaptation du preprocessing pour les conditions opérationnelles multiples
- Le seuil de suppression des capteurs (σ < 0.5) et la fenêtre rolling mean (5 cycles) sont justifiés par l'EDA mais pourraient être optimisés
- Le Random Forest ne modélise pas explicitement les dépendances temporelles longues
- Pas d'optimisation des hyperparamètres — première approche ML classique

---

## Améliorations futures

- Généraliser le pipeline aux datasets FD002, FD003 et FD004
- Optimiser le cap RUL, la fenêtre rolling mean et les hyperparamètres par validation croisée
- Tester des modèles dédiés aux séries temporelles : LSTM, GRU, Transformer
- Ajouter l'importance des variables dans le dashboard
- MLflow pour le suivi des expériences
- Dockeriser l'application et exposer le modèle via FastAPI

---

## Référence

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).  
*Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.*  
PHM'08, Denver, CO. — [PDF](docs/Damage_Propagation_Modeling.pdf)

*Données : [NASA CMAPSS Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) — open data, usage non commercial*