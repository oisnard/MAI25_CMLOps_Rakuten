#!/bin/bash

sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl get pods -n apps -l app=rakuten-api -o wide