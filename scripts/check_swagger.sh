#!/bin/bash

set -e

echo "🔍 Recherche de l'adresse IP locale de la machine hôte..."

SERVICE_IP=$(hostname -I | awk '{print $1}')

if [[ -z "$SERVICE_IP" ]]; then
  echo "❌ Impossible de déterminer l'adresse IP locale."
  exit 1
fi

URL="http://$SERVICE_IP/docs"

echo "🌐 Test d'accès à $URL ..."

if curl -s --head "$URL" | grep "200 OK" > /dev/null; then
  echo "✅ Swagger est accessible à : $URL"
  xdg-open "$URL" >/dev/null 2>&1 || echo "👉 Ouvre manuellement dans ton navigateur : $URL"
else
  echo "⚠️ Impossible d'accéder à Swagger à $URL"
  echo "🧪 Vérifie que le pod est en cours d'exécution et que l'Ingress fonctionne."
fi

for i in {1..10}; do
    if curl -s --fail "$URL" > /dev/null; then
        echo "✅ Swagger est accessible à : $URL"
        exit 0
    else
        echo "⏳ Swagger pas encore dispo, nouvelle tentative dans 3s..."
        sleep 3
    fi
done

echo "❌ Swagger toujours inaccessible après plusieurs tentatives."
exit 1