MAI25_CMLOPS : project Rakuten 
==============================

Projet pédagogique réalisé dans le cadre de la formation MLOps de DataScientest (Cohorte MAI 2025), axé sur la mise en place d’une architecture MLOps complète pour le traitement et la classification de données produits Rakuten dans le cadre du challenge ens-data : https://challengedata.ens.fr/participants/challenges/35/ .
Les modèles déployés sont inspirés de ceux définis par l'équipe Olivier ISNARD / Julien TREVISAN / Loïc RAMAYE lors de leur formation Data Scientist (cohorte Juin 2024) et qui avaient permis d'obtenir la première place au classement public et privé du challenge. Dans le cadre de ce projet, des modèles plus légers ont été mises en place afin de réduire les coûts d'instance AWS (déploiement sur instance EC2 avec GPU).


---
## 🚀 Objectifs réalisés 

- Construire un pipeline ML complet (data loading → preprocessing → entraînement → évaluation → déploiement) avec **Airflow**  
- Suivi des expériences via **MLflow**  
- Suivi du drift des données avec **Evidently**
- Conteneuriser les composants (Docker)  
- Fournir une API REST prédictive sécurisée et déployée sur K8S (avec scalability)
- Versionner les données avec **DVC**  
- Mettre en place des tests unitaires et une intégration continue (CI) via **GitHub Actions** pour garantir la fiabilité du pipeline

---
## 🧭 Schéma d’Architecture MLOps simplifié
![alt text](/data/dataviz/schema_archi.png)

---
## 📁 Structure du dépôt

```bash
.
├── airflow/
│   ├── dags/                  # DAGs Airflow (full pipeline, datastreams, trafic)
│   └── plugins/
├── data/
│   ├── raw/                   # images + CSV ENS Data (DVC)
│   ├── processed/             # jeux train/val/test, features, prédictions (DVC)
│   ├── stream/{raw,processed} # flux simulés
│   ├── monitoring_sample/     # échantillons pour data/drift monitoring
│   └── dataviz/               # schémas & assets pour streamlit
├── docker/ …                  # Dockerfiles (api, airflow, train{_gpu}, preprocess, features{_gpu}, evaluate{_gpu}, mlflow, streamlit, evidently, datastreams, traffic) + prometheus/grafana + requirements
├── k8s/ …                     # manifests K8s (deployment/service/ingress/hpa, PV/PVC, monitoring Prometheus/Grafana, RBAC, templates)
├── src/
│   ├── api/                   # FastAPI (JWT, middleware, service)
│   ├── data/                  # import/make_dataset, preprocessing, datastreams
│   ├── features/              # build_features
│   ├── models/                # train/evaluate/predict (texte, image, fusion)
│   ├── streamlit/             # app Streamlit
│   ├── tools/                 # utilitaires (réseau, datastream)
│   ├── traffic_generation/    # Generation de requêtes API pour tester scale up/down K3S api.
│   └── visualization/         # fonctions de visualisation
├── tests/                     # Tests unitaires pour la CI
├── models/                    # artefacts modèles (poids versionnés via DVC)
├── metrics/                   # rapports de classification (CSV/JSON)
├── monitoring/utils/          # scripts Evidently (drift report)
├── docker-compose.template.yml
├── params.yaml                # Paramètres globaux pour le pipeline (modèle, seed, split, etc.)
├── pytest.ini
├── scripts/                   # CI locale, k3s deploy/cleanup, monitoring, checks
├── generate_compose.sh        # Pour générer le manifeste docker-compose.yml
├── setup.py
├── LICENSE
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
./generate_compose.sh
```
---
### 🧰 Services
| Service     | Port  | Description                            |
| ----------- | ----- | ---------------------------------------|
| Airflow UI  | 8080  | Orchestration du pipeline              |
| MLflow      | 5000  | Tracking des expériences               |
| API FastAPI | 8000  | Endpoint de prédiction                 |
| Evidently   | 9000  | Suivi du data drift                    |
| Prometheus  | 30900 | Monitoring des métriques API           |
| Grafana     | 30300 | Visualisation des métriques Prometheus |
| Streamlit   | 8501  | Application streamlit - Projet         |


---
### ▶️ Lancer l’environnement
1. Prérequis

    Docker installé

    Créer un fichier .env basé sur .env_template. 

2. Lancement des services
    ```
    ./script/deploy_k3s.sh     # Déploiement de l'api sur pod K3S
    ./script/deploy_monitoring.sh # Déploiement du monitoring de l'API via Prometheus/Grafana 
    docker compose up --build  # Lancer les autres services (airflow, mlflow, evidently, streamlit)
    ```
    Airflow sera accessible sur localhost:8080, et MLflow sur localhost:5000.
    Prometheus sera accessible sur http://localhost:90900 (localhost in case of local or EC2 public IP address))
    (Permet de visualiser les métriques exposées par l’API ou Airflow via /metrics)
    Grafana sera accessible sur http://localhost:30300 (Identifiants par défaut : admin / admin) (localhost in case of local or EC2 public IP address)

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
http://localhost:30900
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
http://localhost:30300
```

Identifiants par défaut :
	•	Login : admin
	•	Mot de passe : admin

Pour configurer :
	1.	Aller dans “Connections > Data sources”.
	2.	Cliquer sur “Add data source”.
	3.	Sélectionner Prometheus.
	4.	Renseigner l’URL : http://prometheus:30900.
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

S'assurer d’avoir lancé l'instance EC2 avec GPU, puis exécuter :
```
./scripts/run_ci.sh
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
