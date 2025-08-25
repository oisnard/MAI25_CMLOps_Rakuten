#!/bin/bash

echo "🚀 Déploiement de l'API dans K3s..."


# Charger les variables d'environnement
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "📦 Variables d'environnement chargées depuis .env"
else
    echo "⚠️ Fichier .env introuvable."
    exit 1
fi

# Générer dynamiquement les fichiers PV depuis les templates
echo "📄 Génération des fichiers pv-data.yml et pv-models.yml depuis templates..."
envsubst < k8s/templates/pv-data.template.yml > k8s/pv-data.yml
envsubst < k8s/templates/pv-models.template.yml > k8s/pv-models.yml

# Définir le chemin du KUBECONFIG
KUBECONFIG_FILE="/etc/rancher/k3s/k3s.yaml"

# Vérifier si l'utilisateur peut lire le fichier
if [ ! -r "$KUBECONFIG_FILE" ]; then
    echo "❌ Le fichier $KUBECONFIG_FILE n'est pas lisible. Essayez de relancer ce script avec :"
    echo "   sudo KUBECONFIG=$KUBECONFIG_FILE $0"
    exit 1
fi

# Appliquer le patch du metrics-server
echo "⚙️ Patch du metrics-server..."
kubectl apply -f k8s/metrics-server-patched.yaml

export KUBECONFIG=$KUBECONFIG_FILE

echo "📁 Vérification du namespace 'apps'..."
kubectl get namespace apps >/dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "📁 Le namespace 'apps' n'existe pas. Création..."
  kubectl create namespace apps
else
  echo "✅ Namespace 'apps' déjà existant."
fi

echo "📦 Création des volumes..."
kubectl apply -f k8s/pv-data.yml
kubectl apply -f k8s/pvc-data.yml
kubectl apply -f k8s/pv-models.yml
kubectl apply -f k8s/pvc-models.yml

echo "🚀 Déploiement des composants..."
kubectl apply -f k8s/deployment-api.yml
kubectl apply -f k8s/service-api.yml
kubectl apply -f k8s/ingress-api.yml
kubectl apply -f k8s/hpa-api.yml

echo "⏳ Attente que le pod soit prêt..."
kubectl wait --for=condition=available --timeout=120s deployment/rakuten-api -n apps

echo "✅ Déploiement terminé."