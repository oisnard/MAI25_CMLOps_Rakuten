from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from docker.types import Mount, DeviceRequest
import os
import logging
logger = logging.getLogger(__name__)


BASE_DIR = os.getenv('BASE_DIR')  # Utilise la variable d'environnement si définie
if BASE_DIR is None:
    raise ValueError("La variable d'environnement BASE_DIR est manquante.")
logger.info(f"************** Base directory: {BASE_DIR}")

USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
logger.info(f"************** USE_GPU = {USE_GPU}")


train_kwargs = {}
if USE_GPU:
    train_kwargs["device_requests"] = [
        DeviceRequest(count=-1, capabilities=[["gpu"]])
    ]


# Default arguments for the DAG
# These arguments will be applied to all tasks in the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
    'execution_timeout': timedelta(hours=1),
}
with DAG(
    dag_id='rakuten_datastream_pipeline_dag',
    default_args=default_args,
    description='Pipeline complet pour traiter plusieurs flux de données : data loading, preprocessing, training, évaluation',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    concurrency=1,
    tags=['rakuten', 'mlops', 'datascientest'],
) as dag:

    # Étape 0 : Data Loading
    data_loading = DockerOperator(
        task_id='data_loading_task',
        image='mai25_cmlops_rakuten_dataloading',
        api_version='auto',
        auto_remove=True,
        command='python -m src.data.import_raw_data',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mount_tmp_dir=False,
        mounts=[
            Mount(source=f"{BASE_DIR}/data", target='/app/data', type='bind'),
            Mount(source=f"{BASE_DIR}/src", target='/app/src', type='bind'),
            Mount(source=f"{BASE_DIR}/params.yaml", target='/app/params.yaml', type='bind'),
        ],
    )


    # Étape 1 : Prétraitement
    make_datastream = DockerOperator(
        task_id='makedatastreams_task',
        image='mai25_cmlops_rakuten_make_datastreams',
        api_version='auto',
        auto_remove=True,
        command='python -m src.data.make_datastream',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mount_tmp_dir=False,
        mounts=[
            Mount(source=f"{BASE_DIR}/data", target='/app/data', type='bind'),
            Mount(source=f"{BASE_DIR}/src", target='/app/src', type='bind'),
        ],
    )

    # Étape 2 : Preprocessing des flux de données
    process_datastream = DockerOperator(
        task_id='preprocessing_task',
        image='mai25_cmlops_rakuten_preprocessing_datastreams',
        api_version='auto',
        auto_remove=True,
        command='python -m src.data.process_datastream',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mount_tmp_dir=False,
        mounts=[
            Mount(source=f"{BASE_DIR}/data", target='/app/data', type='bind'),
            Mount(source=f"{BASE_DIR}/src", target='/app/src', type='bind'),
        ],
    )

    # Étape 3 : Entraînement
    train = DockerOperator(
        task_id='training_task',
        image='mai25_cmlops_rakuten_train:latest',
        api_version='auto',
        auto_remove=True,
        command='python -m src.models.train_model_mlflow',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mount_tmp_dir=False,
        mounts=[
            Mount(source=f"{BASE_DIR}/data", target='/app/data', type='bind'),
            Mount(source=f"{BASE_DIR}/models", target='/app/models', type='bind'),
            Mount(source=f"{BASE_DIR}/mlruns", target='/app/mlruns', type='bind'),
            Mount(source=f"{BASE_DIR}/src", target='/app/src', type='bind'),
        ],
        environment={
            "USE_GPU": os.getenv('USE_GPU', 'false'),
        },
        **train_kwargs,
 )

    # Étape 4 : Évaluation
    evaluate = DockerOperator(
        task_id='evaluation_task',
        image='mai25_cmlops_rakuten_evaluate:latest',
        api_version='auto',
        auto_remove=True,
        command='python -m src.models.evaluate_model',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mount_tmp_dir=False,
        mounts=[
            Mount(source=f"{BASE_DIR}/data", target='/app/data', type='bind'),
            Mount(source=f"{BASE_DIR}/models", target='/app/models', type='bind'),
            Mount(source=f"{BASE_DIR}/src", target='/app/src', type='bind'),
        ],
        **train_kwargs,
)


    data_loading >> make_datastream >> process_datastream >> train >> evaluate
