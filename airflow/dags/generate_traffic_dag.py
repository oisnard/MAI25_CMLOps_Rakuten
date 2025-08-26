from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
from docker.types import Mount
import logging
import os 
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
}

BASE_DIR = os.getenv('BASE_DIR')  # Utilise la variable d'environnement si définie
if BASE_DIR is None:
    raise ValueError("La variable d'environnement BASE_DIR est manquante.")
logger.info(f"************** Base directory: {BASE_DIR}")

USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
logger.info(f"************** USE_GPU = {USE_GPU}")

with DAG(
    dag_id='generate_rakuten_traffic',
    default_args=default_args,
    description='DAG to simulate Rakuten API traffic using a Docker container',
    schedule_interval=None,  # Exécution manuelle uniquement
    start_date=datetime(2025, 8, 1),
    catchup=False,
    tags=['rakuten', 'traffic', 'docker'],
) as dag:

    generate_traffic = DockerOperator(
        task_id='generate_rakuten_traffic',
        image='rakuten-traffic-gen',  # nom de l'image construite
        api_version='auto',
        auto_remove=True,
        command='python -m src.traffic_generation.emulate_traffic_V2',
        docker_url='unix://var/run/docker.sock',  # accès local au Docker daemon
        network_mode='host',  # ou 'host' si nécessaire
        environment={
            #"INGRESS_IP": "http://192.168.1.35",
            "INGRESS_IP": "http://172.31.39.207",
            #
        },
        mount_tmp_dir=False,
        mounts=[
#            Mount(source="/etc/rancher/k3s/k3s.yaml", target="/root/.kube/config", type="bind", read_only=True),
            Mount(source=f"{BASE_DIR}/data", target='/app/data', type='bind'),
#            Mount(source="./.env", target="/app/.env", type="bind", read_only=True),
            Mount(source=f"{BASE_DIR}/src", target='/app/src', type='bind'),       
        ],
    )
    generate_traffic