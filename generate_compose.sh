#!/bin/bash

# Lire dynamiquement le GID du socket Docker
DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
export DOCKER_GID

# Charger les variables de .env
set -a
source .env
set +a

# Appliquer la logique conditionnelle GPU
if [ "$USE_GPU" = "true" ]; then
  export DOCKERFILE_TRAIN=docker/Dockerfile.train_gpu
  export DOCKERFILE_EVALUATE=docker/Dockerfile.evaluate_gpu
  export DOCKERFILE_FEATURES=docker/Dockerfile.features_gpu
  export DOCKERFILE_API=docker/Dockerfile.api
else
  export DOCKERFILE_TRAIN=docker/Dockerfile.train
  export DOCKERFILE_EVALUATE=docker/Dockerfile.evaluate
  export DOCKERFILE_FEATURES=docker/Dockerfile.features
  export DOCKERFILE_API=docker/Dockerfile.api
fi

# Générer le docker-compose.yml
envsubst < docker-compose.template.yml > docker-compose.yml
