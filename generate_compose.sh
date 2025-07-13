#!/bin/bash

# Lire dynamiquement le GID du socket Docker
DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)

# Exporter pour que docker-compose puisse l'utiliser
export DOCKER_GID

# Générer le docker-compose.yml
set -a && source .env && set +a && envsubst < docker-compose.template.yml > docker-compose.yml