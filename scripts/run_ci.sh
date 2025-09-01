#!/usr/bin/env bash
set -euo pipefail

# ⚙️ Variables
TEMPLATE="./.github/template/run-tests-ci.template.yml"
TARGET="./.github/workflows/run-tests-ci.yml"

# 1. Vérifier l’accès AWS
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "❌ Erreur : impossible d’appeler AWS CLI. Vérifie tes credentials (aws configure)."
  exit 1
fi

# 2. Récupérer infos de l’unique instance running
read INSTANCE_ID INSTANCE_NAME EC2_IP < <(
  aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].[InstanceId, Tags[?Key=='Name'].Value|[0], PublicIpAddress]" \
    --output text || true
)

if [ -z "${INSTANCE_ID:-}" ] || [ "$INSTANCE_ID" = "None" ] || [ -z "${EC2_IP:-}" ] || [ "$EC2_IP" = "None" ]; then
  echo "❌ Aucune instance EC2 'running' trouvée."
  exit 1
fi

echo "ℹ️ Instance détectée :"
echo "   - ID   : $INSTANCE_ID"
echo "   - Name : $INSTANCE_NAME"
echo "   - IP   : $EC2_IP"

# 3. Générer le workflow depuis le template
sed -e "s/__EC2_IP__/$EC2_IP/g" \
    -e "s/__EC2_NAME__/$INSTANCE_NAME/g" \
    "$TEMPLATE" > "$TARGET"

echo "✅ Fichier $TARGET généré avec succès"

# 4. Générer un nouveau tag run-tests-vXX
latest_tag=$(git tag --list "run-tests-v*" | sort -V | tail -n 1)
if [ -z "$latest_tag" ]; then
  new_tag="run-tests-v1"
else
  version=$(echo "$latest_tag" | grep -oE '[0-9]+$')
  new_version=$((version + 1))
  new_tag="run-tests-v$new_version"
fi

echo "🚀 Nouveau tag : $new_tag"

# 5. Commit + push du workflow mis à jour
git add "$TARGET"
git commit --allow-empty -m "CI: update workflow with EC2 IP $EC2_IP for $new_tag"

current_branch=$(git rev-parse --abbrev-ref HEAD)
git push origin "$current_branch"

# 6. Pousser le tag
git tag "$new_tag"
git push origin "$new_tag"