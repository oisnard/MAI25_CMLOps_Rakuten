#!/bin/bash

set -e

echo "📦 Téléchargement et installation de k3s..."
curl -sfL https://get.k3s.io | sh -

echo "✅ k3s installé avec succès."

echo "🔗 Ajout d’un alias kubectl (pour éviter sudo k3s kubectl)..."
if ! grep -q "alias kubectl=" ~/.bashrc; then
  echo "alias kubectl='sudo k3s kubectl'" >> ~/.bashrc
  echo "✅ Alias ajouté à ~/.bashrc"
else
  echo "ℹ️ Alias kubectl déjà présent dans ~/.bashrc"
fi

echo "🔁 Recharge de ~/.bashrc pour appliquer l'alias..."
source ~/.bashrc

echo "🕒 Patiente pendant que k3s démarre..."
sleep 5

echo "🔍 Vérification du statut du cluster..."
kubectl get nodes

echo "🎉 k3s est installé et prêt à l’emploi !"