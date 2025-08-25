#!/bin/bash

set -e

echo "🚀 Déploiement de la stack de monitoring Prometheus + Grafana..."

# Définir le chemin vers le fichier KUBECONFIG
KUBECONFIG_PATH="/etc/rancher/k3s/k3s.yaml"

# Vérifie si le script est exécuté en tant que root
if [[ "$EUID" -ne 0 ]]; then
  echo "🔒 Droits insuffisants. Relance du script avec sudo..."
  exec sudo -E KUBECONFIG=$KUBECONFIG_PATH "$0" "$@"
fi

export KUBECONFIG=$KUBECONFIG_PATH

# 📡 Récupération de l'IP de la machine
HOST_IP=$(hostname -I | awk '{print $1}')
echo "📡 IP locale détectée : $HOST_IP"

# 🧩 Génération du fichier final à partir du template
PROM_TEMPLATE="k8s/monitoring/prometheus-configmap.template.yaml"
PROM_GENERATED="k8s/monitoring/prometheus-configmap.yaml"

# Remplacer l'IP dans le template
sed "s/REPLACE_WITH_NODE_IP/$HOST_IP/g" $PROM_TEMPLATE > $PROM_GENERATED


# Appliquer le namespace
kubectl apply -f k8s/monitoring/monitoring-namespace.yaml

# Appliquer les ressources RBAC pour Prometheus (accès kubelet)
kubectl apply -f k8s/monitoring/rbac/prometheus-cluster-role.yaml
kubectl apply -f k8s/monitoring/rbac/prometheus-cluster-role-binding.yaml


# Déployer Prometheus
kubectl apply -f "$PROM_GENERATED" #k8s/monitoring/prometheus-configmap.yaml
kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
kubectl apply -f k8s/monitoring/prometheus-service.yaml

# Déployer Grafana
kubectl apply -f k8s/monitoring/grafana/configmaps/grafana-dashboards-configmap.yaml
kubectl apply -f k8s/monitoring/grafana/configmaps/grafana-dashboard-json-configmap.yaml
kubectl apply -f k8s/monitoring/grafana/configmaps/grafana-datasources-configmap.yaml
kubectl apply -f k8s/monitoring/grafana-deployment.yaml
kubectl apply -f k8s/monitoring/grafana-service.yaml


# Déploiement kube-state-metrics
echo "📈 Déploiement de kube-state-metrics..."
kubectl apply -f k8s/monitoring/kube-state-metrics.yaml
kubectl apply -f k8s/monitoring/prometheus-kubelet-rbac.yaml

# Redémarrer Prometheus pour recharger la config
kubectl delete pod -l app=prometheus -n monitoring --ignore-not-found

# Récupération de l'adresse IP locale
IP=$(hostname -I | awk '{print $1}')

echo ""
echo "✅ Monitoring déployé avec succès dans le namespace 'monitoring'."
echo "🔗 Accès aux interfaces web :"
echo "   🔸 Prometheus :   http://$IP:30900"
echo "   🔸 Grafana    :   http://$IP:30300"
echo ""
echo "🔐 Identifiants Grafana (par défaut) :"
echo "   🔸 Utilisateur : admin"
echo "   🔸 Mot de passe : admin"