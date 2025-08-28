MAI25_CMLOPS : project Rakuten 
==============================

Projet pédagogique réalisé dans le cadre de la formation MLOps de DataScientest (Cohorte MAI 2025), axé sur la mise en place d’une architecture MLOps complète pour le traitement et la classification de données produits Rakuten dans le cadre du challenge ens-data : https://challengedata.ens.fr/participants/challenges/35/ .
Les modèles déployés sont dérivés de ceux définis par l'équipe Olivier ISNARD / Julien TREVISAN / Loïc RAMAYE lors de leur formation Data Scientist (cohorte Juin 2025) et qui avaient permis d'obtenir la première place au classement public et privé du challenge. Dans le cadre de ce projet, des modèles plus légers ont été mises en place afin de réduire les coûts d'instance AWS (déploiement sur instance EC2 avec GPU).


---
## 🚀 Objectifs réalisés 

- Mettre en place un pipeline complet de Machine Learning avec Airflow.
- Intégrer des étapes de data loading, preprocessing, entraînement, évaluation et déploiement.
- Suivre les expériences via MLflow.
- Conteneuriser l’environnement avec Docker.
- Fournir une API de prédiction REST sécurisée.
- Suivre les versions de données avec DVC
- Tests unitaires

---
## 🧭 Schéma d’Architecture MLOps simplifié
![alt text](/data/dataviz/schema_archi.png)

---
## 📁 Structure du dépôt

```bash
.
├── airflow/                   # Composants liés à Airflow
│   ├── dags/                 # DAG principal orchestrant le pipeline
│   │   └── rakuten_dags.py
│
├── data/                     # Données versionnées avec DVC
│   ├── raw/                 # Données brutes (ex : images, CSV initiaux)
│   ├── processed/           # Données traitées (X_train, y_train, etc.)
│   ├── processed.dvc        # Fichier DVC de suivi de `/processed`
│   ├── raw.dvc              # Fichier DVC de suivi de `/raw`
│   └── .gitignore           # Évite de traquer les gros fichiers localement
│
├── docker/                   # Dockerfiles spécifiques à chaque étape
│   ├── prometheus.yml        # Configuration du service Prometheus
|   ├── Dockerfile.airflow
│   ├── Dockerfile.api
│   ├── Dockerfile.dataloading
│   ├── Dockerfile.evaluate
│   ├── Dockerfile.mlflow
│   ├── Dockerfile.preprocessing
│   ├── Dockerfile.train
│   ├── requirements-airflow.txt     # Dépendances Airflow
│   ├── requirements-api.txt         # Dépendances FastAPI
│   ├── requirements-dataloading.txt
│   ├── requirements-evaluate.txt
│   ├── requirements-mlflow.txt
│   ├── requirements-preprocessing.txt
│   └── requirements-train.txt
├── docker-compose-template.yml        # Template du docker compose pour générer docker compose selon GPU ou CPU (selon fichier .env)
├── docker-compose.yml                 # Orchestration des services via Docker Compose
│
├── models/                   # Modèles entraînés (.pkl ou autres)
│
├── mlruns/                   # Répertoire d’expérimentation MLflow (tracking local)
│
├── params.yaml               # Paramètres globaux pour le pipeline (modèle, seed, split, etc.)
│
├── src/                      # Code source modulaire pour chaque étape
│   ├── dataloading/         # Scripts pour charger les données brutes
│   ├── preprocessing/       # Feature engineering, normalisation, etc.
│   ├── training/            # Entraînement de modèles
│   ├── evaluation/          # Évaluation de performance
│   └── utils/               # Fonctions utilitaires (log, I/O, etc.)
│
├── tests/                    # Tests unitaires Pytest pour chaque module
│   ├── test_dataloading.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── conftest.py
│
├── .dvc/                     # Répertoire interne de configuration DVC
├── .dvcignore                # Équiv. de .gitignore pour DVC
├── .env                      # Variables d’environnement (ex: BASE_DIR)
└── README.md                 # Documentation du projet (ce fichier)
```
🔍 Fichiers de configuration importants
| Fichier              | Rôle                                                           |
| -------------------- | -------------------------------------------------------------- |
| `docker-compose.yml` | Lance tous les services nécessaires (Airflow, API, MLflow…)    |
| `params.yaml`        | Centralise les hyperparamètres, chemins, splits, etc.          |
| `processed.dvc`      | Suit les transformations de données via DVC                    |
| `.dvcignore`         | Exclut certains fichiers des suivis DVC                        |
| `.env`               | Définit les variables d’environnement Docker (base\_dir, etc.) |
<br>Pour générer le fichier docker-compose.yml à partir du `docker-compose-template.yml`, il faut exécuter la commande : 
```
set -a && source .env && set +a && envsubst < docker-compose.template.yml > docker-compose.yml
```
---
### 🧰 Services
| Service     | Port | Description                            |
| ----------- | ---- | ---------------------------------------|
| Airflow UI  | 8080 | Orchestration du pipeline              |
| MLflow      | 5000 | Tracking des expériences               |
| API FastAPI | 8000 | Endpoint de prédiction                 |
| PostgreSQL  |      | Backend Airflow & MLflow               |
| Redis       |      | Message broker Airflow (Celery)        |
| Prometheus  | 9090 | Monitoring des métriques API           |
| Grafana     | 3000 | Visualisation des métriques Prometheus |

---
### ▶️ Lancer l’environnement
1. Prérequis

    docker installé

    Un fichier .env avec les variables suivantes : 
    ```
    # Description: Environment variables for the Rakuten project
    # The account must be subscribed to the challenge https://challengedata.ens.fr/participants/challenges/35/
    ENSDATA_LOGIN=
    ENSDATA_PASSWORD=

    # The path to the directory where the data is stored
    DATA_RAW_DIR="./data/raw"
    # The path to the directory where the the images of train dataset are stored
    DATA_RAW_IMAGES_TRAIN_DIR="./data/raw/image_train"
    # The path to the directory where the the images of test dataset are stored
    DATA_RAW_IMAGES_TEST_DIR="./data/raw/image_test"

    # The path to the directory where the processed data will be stored
    DATA_PROCESSED_DIR="./data/processed"
    # The path to the directory where the model will be stored
    MODEL_DIR="./models"
    # The path to the directory where the logs will be stored
    LOGS_DIR="./logs"
    # The path to the directory where the scores of model evaluation will be stored
    METRICS_DIR="./metrics"

    # Definition of the secret key for signing JWT tokens
    # This key should be kept secret and not shared publicly
    # It is used to ensure the integrity and authenticity of the JWT tokens
    # It is recommended to use a strong, random key for production environments

    JWT_SECRET_KEY = 

    FERNET_KEY=

    # Local directory where is stored the projet
    BASE_DIR = 

    # The Dockerfile to use for training the model
    # If GPU is available, then set Dockerfile.evaluate_gpu
    DOCKERFILE_TRAIN=docker/Dockerfile.train

    # The Dockerfile to use for evaluating the model
    # If GPU is available, then set Dockerfile.evaluate_gpu
    DOCKERFILE_EVALUATE=docker/Dockerfile.evaluate

    # The Dockerfile to use for launching API rest with the model
    # If GPU is available, then set Dockerfile.api_gpu
    DOCKERFILE_API=docker/Dockerfile.api
    ```
2. Lancement des services
    ```
    docker compose up --build
    ```
    Airflow sera accessible sur localhost:8080, et MLflow sur localhost:5000.
    Prometheus sera accessible sur http://localhost:9090
    (Permet de visualiser les métriques exposées par l’API ou Airflow via /metrics)
    Grafana sera accessible sur http://localhost:3000 (Identifiants par défaut : admin / admin)

---
### ⚙️ Pipelines Airflow

Le DAG principal (rakuten_dags.py) orchestre les étapes suivantes :
- data_loading_task
- preprocessing_task
- training_task
- evaluation_task
- mlflow_dag_task

---
### 🧪 Suivi des expériences

Les logs et artefacts d'entraînement sont automatiquement tracés dans MLflow :

```http://localhost:5000 ```

Le tracking inclut :
- métriques d’évaluation
- paramètres du modèle
- visualisations

---
### 🌐 API de prédiction

Un service FastAPI expose un endpoint sur le port 8000.

---
### 🧪 Tests et reproductibilité

Les composants sont encapsulés dans des images Docker distinctes pour chaque étape (dataloading, preprocessing, etc.), facilitant l’isolation et les tests.

---
### 🗂️ Gestion des données avec DVC

Les transformations sont suivies avec DVC pour permettre une versioning des datasets transformés.
```
dvc repro
dvc push
``` 

---
### 📊 Monitoring avec Prometheus et Grafana
L’API FastAPI est instrumentée avec `prometheus_fastapi_instrumentator` pour exposer des métriques accessibles sur :

```
http://localhost:8000/metrics
```

🔎 Prometheus

Prometheus collecte les métriques de l’API toutes les 15 secondes.
Interface accessible via :

```
http://localhost:9090
```

Exemple de requête PromQL à exécuter dans l’interface :

```
sum by (handler) (http_requests_total)
```

Cela permet de visualiser le nombre total de requêtes par endpoint.


📈 Grafana

Grafana permet de créer des dashboards personnalisés à partir des données Prometheus.
Accès à Grafana :

```
http://localhost:3000
```

Identifiants par défaut :
	•	Login : admin
	•	Mot de passe : admin

Pour configurer :
	1.	Aller dans “Connections > Data sources”.
	2.	Cliquer sur “Add data source”.
	3.	Sélectionner Prometheus.
	4.	Renseigner l’URL : http://prometheus:9090.
	5.	Créer des panels à partir de requêtes PromQL (ex: http_requests_total).

---
### ✅ Tests unitaires

Le répertoire tests/ contient des tests unitaires pour valider les différentes étapes du pipeline ML : chargement des données, prétraitement, entraînement, prédiction, etc.
📦 Structure
```
tests/
├── test_dataloading.py
├── test_preprocessing.py
├── test_training.py
├── test_evaluation.py
└── conftest.py  # Fixtures partagées 
```
▶️ Exécution des tests

Assure-toi d’avoir installé pytest (via pip install pytest ou via un requirements.txt), puis lance les tests avec :
```
python -m pytest
```

### Installation k3s
Lancer le script :
```
./scripts/install_k3s.sh
```

### Déployer l'api sur k3s et le monitoring
Lancer les scripts
```
./scripts/deploy_k3s.sh
./scripts/deploy_monitoring.sh
```

### Nettoyer les ressources k3s (namespace, pv/pvc, pods)
Lancer les scripts
```
./scripts/cleanup_monitoring.sh
./scripts/cleanup_k3s.sh
```

---
### 📝 Auteurs

- Olivier ISNARD
- Christian SEGNOU

Encadrés dans le cadre de la formation MLOps par Maria de DataScientest.
