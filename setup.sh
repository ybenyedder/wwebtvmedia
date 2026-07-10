#!/usr/bin/env bash
#
# Installe wwebtvmedia dans un environnement virtuel Python et configure
# l'accès au LLM (clé d'API + URL de base), écrit dans un fichier .env.
#
#   ./setup.sh
#
# Variables optionnelles :
#   PYTHON=python3.12   interpréteur à utiliser
#   WWEBTV_VENV=.venv   répertoire de l'environnement virtuel
#   WWEBTV_SYSTEM_SITE=1  réutilise les paquets système (évite de réinstaller torch)
#   WWEBTV_SKIP_INSTALL=1 saute l'installation des dépendances (config seule)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
VENV="${WWEBTV_VENV:-.venv}"
ENV_FILE=".env"

# --- 1. Environnement virtuel ---
echo "== 1/3  Environnement virtuel ($VENV) =="
if [ ! -d "$VENV" ]; then
    VENV_ARGS=()
    [ "${WWEBTV_SYSTEM_SITE:-0}" = "1" ] && VENV_ARGS+=(--system-site-packages)
    "$PYTHON" -m venv "${VENV_ARGS[@]}" "$VENV"
    echo "   créé."
else
    echo "   déjà présent."
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- 2. Dépendances ---
echo "== 2/3  Dépendances (requirements.txt) =="
if [ "${WWEBTV_SKIP_INSTALL:-0}" = "1" ]; then
    echo "   ignoré (WWEBTV_SKIP_INSTALL=1)."
else
    python -m pip install --upgrade pip >/dev/null
    python -m pip install -r requirements.txt
fi

# --- 3. Configuration du LLM (clé + URL) ---
echo "== 3/3  Configuration du LLM =="
read -r -p "Fournisseur LLM [anthropic/deepseek] (défaut: anthropic) : " PROVIDER
PROVIDER="${PROVIDER:-anthropic}"

read -r -s -p "Clé d'API du LLM (laisser vide pour configurer plus tard) : " API_KEY
echo

if [ "$PROVIDER" = "deepseek" ]; then
    DEFAULT_URL="https://api.deepseek.com"
else
    DEFAULT_URL="https://api.anthropic.com"
fi
read -r -p "URL de base du LLM (défaut: $DEFAULT_URL) : " BASE_URL
BASE_URL="${BASE_URL:-$DEFAULT_URL}"

{
    echo "# Généré par setup.sh — NE PAS committer (contient la clé d'API)."
    echo "LLM_PROVIDER=$PROVIDER"
    if [ "$PROVIDER" = "deepseek" ]; then
        [ -n "$API_KEY" ] && echo "DEEPSEEK_API_KEY=$API_KEY"
        echo "DEEPSEEK_BASE_URL=$BASE_URL"
    else
        [ -n "$API_KEY" ] && echo "ANTHROPIC_API_KEY=$API_KEY"
        echo "ANTHROPIC_BASE_URL=$BASE_URL"
    fi
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"
echo "   $ENV_FILE écrit (fournisseur : $PROVIDER, URL : $BASE_URL)."

cat <<EOF

Installation terminée.

  Activer l'environnement :  source $VENV/bin/activate
  Les scripts chargent .env automatiquement (clé + URL du LLM).

  Exemples :
    python generate_dataset.py --provider $PROVIDER --num 100
    python svg_fit.py --from-llm-svg "un soleil sur des collines" --provider $PROVIDER
    python main.py --help
EOF
