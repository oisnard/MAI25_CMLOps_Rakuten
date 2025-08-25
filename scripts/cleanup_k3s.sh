#!/bin/bash

#!/bin/bash

echo "🧼 Suppression namespace apps..."
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl delete namespace apps --ignore-not-found

echo "🧼 Suppression des PVCs.."
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl delete -f k8s/pvc-data.yml -n apps --ignore-not-found
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl delete -f k8s/pvc-models.yml -n apps --ignore-not-found

echo "🧼 Suppression des PV..."
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl delete pv pv-data --ignore-not-found
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl delete pv pv-models --ignore-not-found

echo "🧼 Suppression du patch metrics-server..."
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl delete -f k8s/metrics-server-patched.yaml --ignore-not-found

echo "🧼 Suppression de l'Ingress..."
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl delete -f k8s/ingress-api.yml -n apps --ignore-not-found



echo "✅ Cleanup terminé."

