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


# Lire dynamiquement le GID de l'utilisateur courant
USER_GID=$(id -g)

# Paramètre pour lancer le service API
WITH_API=false
if [ "$1" == "--with-api" ]; then
  WITH_API=true
fi

# Substitution du GID
sed -i "s/{USER_GID}/$USER_GID/g" docker-compose.yml

if $WITH_API; then
  echo "Activation du service 'api' et désactivation de 'traffic-generator'..."

  echo "Activation de l'API : on commente 'profiles: [\"buildonly\"]'"
  sed -i '/^[[:space:]]*profiles: \["buildonly"\]/ s/^/#/' docker-compose.yml


  # Commente le bloc 'traffic-generator'
  awk '
    BEGIN {in_block=0}
    /^[[:space:]]*traffic-generator:/ {in_block=1}
    in_block && /^[^[:space:]#]/ && !/^traffic-generator:/ {in_block=0}
    in_block {print "#" $0; next}
    {print $0}
  ' docker-compose.yml > tmp && mv tmp docker-compose.yml
fi

echo "Fichier docker-compose.yml généré avec succès."