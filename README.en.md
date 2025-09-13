MAI25_CMLOPS: Rakuten Project 
==============================

Educational project carried out as part of the **MLOps training** at DataScientest (Cohort MAI 2025), focused on implementing a complete MLOps architecture for processing and classifying Rakuten product data in the context of the ENS Data Challenge: https://challengedata.ens.fr/participants/challenges/35/.  
The deployed models are inspired by those defined by the team Olivier ISNARD / Julien TREVISAN / Loïc RAMAYE during their Data Scientist training (June 2024 cohort), which achieved 1st place in both the public and private leaderboards of the challenge.  
In this project, lighter models were implemented in order to reduce AWS instance costs (deployment on EC2 with GPU).  

---
## 🚀 Objectives achieved

- Build a complete ML pipeline (data loading → preprocessing → training → evaluation → deployment) with **Airflow**  
- Track experiments with **MLflow**  
- Monitor data drift with **Evidently**  
- Containerize components with **Docker**  
- Provide a secure predictive REST API deployed on **Kubernetes (K8s)** with scalability  
- Version datasets with **DVC**  
- Implement unit tests and **continuous integration (CI)** with **GitHub Actions** to ensure pipeline reliability  

---
## 🧭 Simplified MLOps Architecture Diagram
![alt text](/data/dataviz/schema_archi.png)

---
## 📁 Repository structure

```bash
.
├── airflow/
│   ├── dags/                  # Airflow DAGs (full pipeline, datastreams, traffic)
│   └── plugins/
├── data/
│   ├── raw/                   # ENS Data CSVs + images (DVC)
│   ├── processed/             # train/val/test sets, features, predictions (DVC)
│   ├── stream/{raw,processed} # simulated streams
│   ├── monitoring_sample/     # samples for data/drift monitoring
│   └── dataviz/               # diagrams & assets for streamlit
├── docker/ …                  # Dockerfiles (api, airflow, train{_gpu}, preprocess, features{_gpu}, evaluate{_gpu}, mlflow, streamlit, evidently, datastreams, traffic) + prometheus/grafana + requirements
├── k8s/ …                     # K8s manifests (deployment/service/ingress/hpa, PV/PVC, Prometheus/Grafana monitoring, RBAC, templates)
├── src/
│   ├── api/                   # FastAPI (JWT, middleware, service)
│   ├── data/                  # import/make_dataset, preprocessing, datastreams
│   ├── features/              # build_features
│   ├── models/                # train/evaluate/predict (text, image, fusion)
│   ├── streamlit/             # Streamlit app
│   ├── tools/                 # utilities (network, datastream)
│   ├── traffic_generation/    # API request generator for K3S scaling tests
│   └── visualization/         # visualization functions
├── tests/                     # Unit tests for CI
├── models/                    # model artifacts (weights tracked via DVC)
├── metrics/                   # classification reports (CSV/JSON)
├── monitoring/utils/          # Evidently scripts (drift reports)
├── docker-compose.template.yml
├── params.yaml                # Global pipeline parameters (model, seed, split, etc.)
├── pytest.ini
├── scripts/                   # Local CI, k3s deploy/cleanup, monitoring, checks
├── generate_compose.sh        # Generate docker-compose.yml from template
├── setup.py
├── LICENSE
├── .env                       # Environment variables (ex: BASE_DIR)
└── README.md                  # Project documentation (French version)
```

🔍 Key configuration files
| File                | Role                                                              |
| ------------------- | ----------------------------------------------------------------- |
| `docker-compose.yml`| Launches all required services (Airflow, API, MLflow, etc.)       |
| `params.yaml`       | Centralizes hyperparameters, paths, splits, etc.                  |
| `processed.dvc`     | Tracks data transformations via DVC                               |
| `.dvcignore`        | Excludes files from DVC tracking                                  |
| `.env`              | Defines Docker environment variables (base_dir, etc.)             |

<br>To generate the `docker-compose.yml` file from the `docker-compose-template.yml`, run:
```
./generate_compose.sh
```

---
### 🧰 Services
| Service     | Port  | Description                           |
| ----------- | ----- | ------------------------------------- |
| Airflow UI  | 8080  | Pipeline orchestration                |
| MLflow      | 5000  | Experiment tracking                   |
| API FastAPI | 8000  | Prediction endpoint                   |
| Evidently   | 9000  | Data drift monitoring                 |
| Prometheus  | 90900 | API metrics monitoring                |
| Grafana     | 30300 | Prometheus metrics visualization      |
| Streamlit   | 8501  | Streamlit project app                 |

---
### ▶️ Start the environment
1. Prerequisites

   - Docker installed  
   - Create a `.env` file based on `.env_template`  

2. Launch the services
   ```
   ./scripts/deploy_k3s.sh        # Deploy API on K3S pod
   ./scripts/deploy_monitoring.sh # Deploy API monitoring with Prometheus/Grafana
   docker compose up --build      # Launch other services (Airflow, MLflow, Evidently, Streamlit)
   ```

   - Airflow → [http://localhost:8080](http://localhost:8080)  
   - MLflow → [http://localhost:5000](http://localhost:5000)  
   - Prometheus → [http://localhost:90900](http://localhost:90900) (metrics from API & Airflow via `/metrics`)  
   - Grafana → [http://localhost:30300](http://localhost:30300) (default login: `admin / admin`)  

---
### ⚙️ Airflow Pipelines

The main DAG (`rakuten_dags.py`) orchestrates the following steps:
- data_loading_task  
- preprocessing_task  
- training_task  
- evaluation_task  
- mlflow_dag_task  

---
### 🧪 Experiment Tracking

Training logs and artifacts are automatically tracked in MLflow:

```http://localhost:5000 ```

Tracking includes:
- evaluation metrics  
- model parameters  
- visualizations  

---
### 🌐 Prediction API

A FastAPI service exposes a prediction endpoint on port **8000**.

---
### 🧪 Testing & Reproducibility

Each component is encapsulated in a dedicated Docker image (dataloading, preprocessing, etc.), ensuring isolation and reproducibility.

---
### 🗂️ Data management with DVC

Data transformations are tracked with **DVC**, enabling dataset versioning:
```
dvc repro
dvc push
``` 

---
### 📊 Monitoring with Prometheus & Grafana

The FastAPI app is instrumented with `prometheus_fastapi_instrumentator` to expose metrics:

```
http://localhost:8000/metrics
```

🔎 Prometheus

Prometheus scrapes API metrics every 15 seconds.  
UI accessible at:

```
http://localhost:90900
```

Example PromQL query:

```
sum by (handler) (http_requests_total)
```

This shows the total number of requests per endpoint.

📈 Grafana

Grafana allows dashboards creation from Prometheus data.  
Access:

```
http://localhost:30300
```

Default credentials:  
- Login: `admin`  
- Password: `admin`  

To configure:  
1. Go to “Connections > Data sources”  
2. Click “Add data source”  
3. Select **Prometheus**  
4. Set URL: `http://prometheus:90900`  
5. Create panels with PromQL queries (e.g., `http_requests_total`)  

---
### ✅ Unit tests

The `tests/` directory contains unit tests to validate pipeline steps: data loading, preprocessing, training, prediction, etc.  

📦 Structure
```
tests/
├── test_dataloading.py
├── test_preprocessing.py
├── test_training.py
├── test_evaluation.py
└── conftest.py  # Shared fixtures
```

▶️ Run tests

Make sure your EC2 instance with GPU is running, then execute:
```
./scripts/run_ci.sh
```

---
### K3s installation
Run:
```
./scripts/install_k3s.sh
```

### Deploy API & monitoring on K3s
Run:
```
./scripts/deploy_k3s.sh
./scripts/deploy_monitoring.sh
```

### Clean up K3s resources (namespace, pv/pvc, pods)
Run:
```
./scripts/cleanup_monitoring.sh
./scripts/cleanup_k3s.sh
```

---
### 📝 Authors

- Olivier ISNARD  
- Christian SEGNOU  

Supervised as part of the MLOps program by Maria (DataScientest).
