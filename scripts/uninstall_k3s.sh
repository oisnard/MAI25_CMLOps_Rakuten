#!/bin/bash

set -e

echo "🧹 Lancement de la désinstallation de k3s..."

# 1. Exécute le script officiel de désinstallation
if [ -f /usr/local/bin/k3s-uninstall.sh ]; then
  sudo /usr/local/bin/k3s-uninstall.sh
else
  echo "⚠️ Script de désinstallation introuvable à /usr/local/bin/k3s-uninstall.sh"
  echo "🔍 Tentative de suppression manuelle..."
  sudo systemctl stop k3s || true
  sudo systemctl disable k3s || true
  sudo rm -rf /etc/rancher/k3s
  sudo rm -f /usr/local/bin/k3s
  sudo rm -f /etc/systemd/system/k3s.service
fi

# 2. Supprimer les résidus éventuels
sudo rm -rf /var/lib/rancher /var/lib/kubelet /etc/rancher /var/lib/cni /opt/cni

# 3. Nettoyer l'alias dans ~/.bashrc s’il existe
if grep -q "alias kubectl='sudo k3s kubectl'" ~/.bashrc; then
  echo "🧽 Suppression de l'alias kubectl de ~/.bashrc..."
  sed -i "/alias kubectl='sudo k3s kubectl'/d" ~/.bashrc
fi

# 4. Recharge du bashrc
source ~/.bashrc

echo "✅ k3s a été désinstallé avec succès."