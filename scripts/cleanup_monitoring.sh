#!/bin/bash

# Définir le chemin vers le fichier KUBECONFIG
KUBECONFIG_PATH="/etc/rancher/k3s/k3s.yaml"

# Vérifie si le script est exécuté en tant que root
if [[ "$EUID" -ne 0 ]]; then
  echo "🔒 Droits insuffisants. Relance du script avec sudo..."
  exec sudo -E KUBECONFIG=$KUBECONFIG_PATH "$0" "$@"
fi

echo "🧹 Suppression des ressources de monitoring Prometheus + Grafana..."

# Namespace
kubectl delete -f k8s/monitoring/monitoring-namespace.yaml --ignore-not-found

# Grafana
kubectl delete -f k8s/monitoring/grafana-service.yaml --ignore-not-found
kubectl delete -f k8s/monitoring/grafana-deployment.yaml --ignore-not-found
kubectl delete -f k8s/monitoring/grafana/configmaps/grafana-datasources-configmap.yaml --ignore-not-found
kubectl delete -f k8s/monitoring/grafana/configmaps/grafana-dashboards-configmap.yaml --ignore-not-found
kubectl delete -f k8s/monitoring/grafana/configmaps/grafana-dashboard-json-configmap.yaml --ignore-not-found

# Prometheus
kubectl delete -f k8s/monitoring/prometheus-service.yaml --ignore-not-found
kubectl delete -f k8s/monitoring/prometheus-deployment.yaml --ignore-not-found
kubectl delete -f k8s/monitoring/prometheus-configmap.yaml --ignore-not-found

# Supprimer les RBAC
kubectl delete -f k8s/monitoring/rbac/prometheus-cluster-role.yaml --ignore-not-found
kubectl delete -f k8s/monitoring/rbac/prometheus-cluster-role-binding.yaml --ignore-not-found



# Kube-state-metrics
kubectl delete -f k8s/monitoring/kube-state-metrics.yaml --ignore-not-found
kubectl delete -f k8s/monitoring/prometheus-kubelet-rbac.yaml --ignore-not-found

rm k8s/monitoring/prometheus-configmap.yaml

echo "✅ Ressources de monitoring supprimées avec succès."