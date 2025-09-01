#!/usr/bin/env bash
set -euo pipefail

# ⚙️ Variables
TEMPLATE="./.github/workflows/run-tests-ci.template.yml"
TARGET="./.github/workflows/run-tests-ci.yml"

# 1. Vérifier qu'AWS CLI fonctionne
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "❌ Erreur : impossible d’appeler AWS CLI. Vérifie tes credentials (aws configure)."
  exit 1
fi

# 2. Récupérer infos de l'unique instance en "running"
read INSTANCE_ID INSTANCE_NAME EC2_IP < <(
  aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].[InstanceId, Tags[?Key=='Name'].Value|[0], PublicIpAddress]" \
    --output text || true
)

if [ -z "${INSTANCE_ID:-}" ] || [ "$INSTANCE_ID" = "None" ] || [ -z "${EC2_IP:-}" ] || [ "$EC2_IP" = "None" ]; then
  echo "❌ Aucune instance EC2 'running' trouvée dans cette région."
  exit 1
fi

echo "ℹ️ Instance détectée :"
echo "   - ID     : $INSTANCE_ID"
echo "   - Name   : $INSTANCE_NAME"
echo "   - IP     : $EC2_IP"

# 3. Générer le fichier YAML depuis le template (local uniquement)
sed -e "s/__EC2_IP__/$EC2_IP/g" \
    -e "s/__EC2_NAME__/$INSTANCE_NAME/g" \
    "$TEMPLATE" > "$TARGET"

echo "✅ Fichier $TARGET généré avec l’IP $EC2_IP"

# 4. Générer un nouveau tag run-tests-vXX
latest_tag=$(git tag --list "run-tests-v*" | sort -V | tail -n 1)
if [ -z "$latest_tag" ]; then
  new_tag="run-tests-v1"
else
  version=$(echo "$latest_tag" | grep -oE '[0-9]+$')
  new_version=$((version + 1))
  new_tag="run-tests-v$new_version"
fi

echo "🚀 Nouveau tag généré : $new_tag"

# 5. Pousser uniquement le tag (pas de commit du workflow)
git tag "$new_tag"
git push origin "$new_tag"

# 6. Nettoyer le fichier généré localement
rm "$TARGET"
echo "🧹 Fichier $TARGET supprimé (non commité)"