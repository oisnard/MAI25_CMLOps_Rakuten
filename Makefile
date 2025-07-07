# Makefile pour MAI25_CMLOps_Rakuten
PROJECT_NAME=rakuten_pipeline
AIRFLOW_UID=50000
COMPOSE=docker compose

.PHONY: help build up down restart logs init dags clean

help:
	@echo "Commandes disponibles :"
	@echo "  make build     - Build les images Docker personnalisées"
	@echo "  make up        - Lance tous les services Airflow et MLFlow"
	@echo "  make down      - Arrête tous les services"
	@echo "  make restart   - Redémarre tous les services"
	@echo "  make logs      - Affiche les logs d'Airflow"
	@echo "  make init      - Initialise la base Airflow (utilise airflow-init)"
	@echo "  make dags      - Liste les DAGs chargés"
	@echo "  make clean     - Supprime volumes, DB, logs et reconstruit"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart: down up

logs:
	$(COMPOSE) logs -f airflow-webserver

init:
	$(COMPOSE) run --rm airflow-init

dags:
	$(COMPOSE) exec airflow-webserver airflow dags list

clean:
	$(COMPOSE) down -v
	rm -rf ./airflow/logs/*
	rm -rf ./airflow/db/*
	$(COMPOSE) build
	$(COMPOSE) run --rm airflow-init