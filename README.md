# wwebtvmedia

Deux générateurs pilotés par un **prompt texte**, entraînables sur une seule
machine (GPU recommandé pour les images) :

1. **Images** — VAE conditionné par prompt + flow matching (rectified flow)
   dans l'espace latent, entraîné sur CIFAR-10 avec des captions synthétiques
   dérivées des 10 classes. Le générateur complet (encodeur de texte +
   intégration RK4 + décodeur) s'exporte en **un seul graphe ONNX** à batch
   dynamique.
2. **Code** — transformer décodeur byte-level entraîné sur des paires
   `(prompt, code)` au format JSONL. La perte n'est appliquée que sur la
   partie code : le modèle apprend à générer du code, pas à recopier les
   prompts.

## Installation

```bash
git clone https://github.com/ybenyedder/wwebtvmedia.git
cd wwebtvmedia
./setup.sh                 # crée .venv, installe tout, demande la clé + l'URL du LLM
source .venv/bin/activate
```

### Script d'installation (recommandé)

`setup.sh` crée un environnement virtuel, installe toutes les dépendances, puis
demande la **clé d'API** et l'**URL** du LLM et les enregistre dans un fichier
`.env` :

```bash
./setup.sh
source .venv/bin/activate
```

Le script demande le fournisseur (`anthropic` ou `deepseek`), la clé d'API et
l'URL de base, et écrit `.env` (permissions `600`, ignoré par git). Options :

```bash
WWEBTV_SYSTEM_SITE=1 ./setup.sh   # réutilise les paquets système (ex. torch déjà installé)
WWEBTV_VENV=env ./setup.sh        # nom de l'environnement virtuel
```

### Installation manuelle

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # puis renseignez la clé et l'URL du LLM
```

### Configuration du LLM (`.env`)

Les scripts qui appellent un LLM (`generate_dataset.py`, `svg_fit.py
--from-llm-svg`) chargent `.env` **automatiquement** au démarrage. Variables
reconnues :

| Variable | Rôle |
|---|---|
| `LLM_PROVIDER` | fournisseur par défaut : `anthropic` ou `deepseek` |
| `ANTHROPIC_API_KEY` | clé d'API Claude |
| `ANTHROPIC_BASE_URL` | URL de base Claude (optionnelle) |
| `DEEPSEEK_API_KEY` | clé d'API DeepSeek |
| `DEEPSEEK_BASE_URL` | URL de base DeepSeek (défaut `https://api.deepseek.com`) |

Voir `.env.example` pour un modèle prêt à copier.

## Utilisation

```bash
# 1. Images : phase 1 (VAE + encodeur de texte) puis phase 2 (flow matching),
#    avec reprise automatique sur checkpoint, export ONNX et grille d'exemples
python main.py train-image

# 2. Code : entraînement sur les paires (prompt, code)
python main.py train-code --data data/code_pairs.jsonl --epochs 100

# Génération
python main.py generate-image --prompt "une photo de chat" --n 8
python main.py generate-image --prompt "une photo de chat" --prompt "un camion"
python main.py generate-code  --prompt "écris une fonction fibonacci"

# Ré-export ONNX seul (après entraînement)
python main.py export-onnx           # générateur d'images
python main.py export-code-onnx      # générateur de code (avec cache KV)

# Génération de code via le graphe ONNX + cache KV (sous onnxruntime)
python main.py generate-code-onnx --prompt "écris une fonction fibonacci" --seed 0
```

## Données d'entraînement du générateur de code

Un fichier JSONL, un exemple par ligne :

```json
{"prompt": "écris une fonction qui inverse une chaîne", "code": "def inverser(s):\n    return s[::-1]"}
```

Le fichier `data/code_pairs.jsonl` fourni (30 exemples) sert uniquement à
valider la chaîne complète. Remplacez-le par votre propre corpus — des
milliers de paires minimum pour un début de généralisation.

### Générer les vecteurs automatiquement via un LLM (`generate_dataset.py`)

Plutôt que de rédiger les paires à la main, `generate_dataset.py` les fait
produire par un LLM, en deux phases : le modèle génère d'abord des **prompts**
(énoncés de tâches variés, par catégorie), puis le **code** correspondant à
chaque prompt. La sortie est écrite dans le fichier lu par `main.py` et validée
avec son propre `CodePairDataset`.

Deux fournisseurs sont pris en charge via `--provider` :

- **`anthropic`** (défaut) — API Claude, sorties structurées natives.
  Clé : `ANTHROPIC_API_KEY` (ou `ant auth login`). Modèle : `claude-opus-4-8`.
- **`deepseek`** — API DeepSeek (compatible OpenAI), mode JSON.
  Clé : `DEEPSEEK_API_KEY`. Modèle : `deepseek-chat`.

```bash
# Claude (défaut)
export ANTHROPIC_API_KEY=sk-ant-...          # ou : ant auth login
python generate_dataset.py --num 300         # ajoute 300 paires à data/code_pairs.jsonl

# DeepSeek
export DEEPSEEK_API_KEY=sk-...
python generate_dataset.py --provider deepseek --num 300

# Autres options
python generate_dataset.py --num 40 --language "JavaScript" --output data/js_pairs.jsonl
python generate_dataset.py --dry-run --num 20   # teste toute la chaîne SANS API ni clé
python main.py train-code --data data/code_pairs.jsonl   # entraîne sur les vecteurs produits
```

Points clés :

- **Deux fournisseurs interchangeables** (Claude, DeepSeek) derrière une seule
  interface — même format de sortie quel que soit le fournisseur.

- **Connecté dynamiquement à `main.py`** : même fichier de sortie (`CODE_DATA`),
  respect de la contrainte de longueur (`MAX_CODE_LEN`), validation finale via
  `CodePairDataset`.
- **Sorties structurées** (schéma Pydantic) : sorties structurées natives côté
  Claude, mode JSON côté DeepSeek, avec parsing tolérant (retrait des balises
  ``` et des sauts de ligne littéraux que certains modèles émettent).
- **Reprise et déduplication** : `--append` (défaut) reprend le fichier existant
  et ignore les prompts déjà présents ; écriture ligne par ligne (un crash
  conserve le progrès).
- **Robustesse** : nouvelles tentatives automatiques du SDK sur 429/5xx, un échec
  de génération n'interrompt pas le lot, les paires trop longues sont écartées
  (comptées, pas tronquées silencieusement).
- **`--dry-run`** : générateur factice hors-ligne pour valider la chaîne complète
  sans clé d'API.

Modèle par défaut : `claude-opus-4-8` (anthropic) ou `deepseek-chat` (deepseek),
modifiable avec `--model`.

## Détails techniques

- Tokenisation **byte-level** partagée (aucun vocabulaire à télécharger),
  tokens spéciaux `PAD/BOS/SEP/EOS`.
- L'encodeur de texte (transformer 2 couches) est entraîné conjointement avec
  le VAE en phase 1, puis gelé pour la phase 2.
- Le générateur de code est un transformer décodeur maison avec **cache KV** :
  au lieu de recalculer l'attention sur tout le préfixe à chaque token,
  `sample_code` amorce le cache une fois sur le prompt puis n'encode que le
  nouveau token à chaque pas — décodage en O(n) au lieu de O(n²). L'équivalence
  exacte avec le recalcul complet (écart ~1e-6) est vérifiée par les tests.
- **Export ONNX du générateur de code avec cache KV** (`export-code-onnx`) : le
  graphe est un *pas de décodage* qui expose le cache en entrées (`past_k/v_i`)
  et sorties (`present_k/v_i`) par couche, avec `input_ids`, `position_ids` et
  un masque additif. Un seul graphe gère l'amorçage (past de longueur 0) et les
  pas incrémentaux (une seule position), grâce aux axes dynamiques. L'attention
  est recalculée explicitement (matmul + softmax) pour un export robuste. La
  génération sous onnxruntime (`generate-code-onnx`) est vérifiée **identique**
  au décodage PyTorch en greedy.
- Les latents sont **normalisés par canal** (statistiques stockées dans le
  checkpoint) avant le flow matching, pour partir d'un bruit N(0,1) cohérent.
- **EMA** (decay 0,999) sur le réseau de drift, utilisée pour l'export et la
  génération.
- Checkpoints à écriture **atomique**, reprise automatique phase/époque.
- Export ONNX (opset 18) vérifié par comparaison PyTorch/onnxruntime en
  batch > 1.

## Support SVG (`svg_fit.py`)

Vectorisation d'image en **SVG** par **rendu différentiable + distance à
l'image** (approche « differentiable vector graphics », en PyTorch pur, sans
diffvg) :

1. une image SVG est paramétrée par N ellipses colorées (centre, rayons,
   rotation, couleur, opacité) + un fond ;
2. un **rasteriseur différentiable** rend ces primitives en pixels (couverture
   douce + compositing source-over), donc les gradients traversent le rendu ;
3. on **entraîne** les primitives pour **minimiser la distance (MSE)** avec une
   image cible ;
4. on exporte un vrai fichier `.svg` (ouvrable dans un navigateur) + un aperçu
   PNG du rendu.

La cible peut être :

```bash
# Cible de démonstration synthétique
python svg_fit.py --shapes 60 --steps 500

# Un fichier image quelconque
python svg_fit.py --target photo.png --shapes 80 --out out.svg

# Une image produite par le générateur du pipeline (« l'image générée »)
python svg_fit.py --from-prompt "une photo de chat" --shapes 60

# Un SVG GÉNÉRÉ PAR UN LLM (Claude ou DeepSeek), rasterisé puis appris
export DEEPSEEK_API_KEY=sk-...
python svg_fit.py --from-llm-svg "un soleil au-dessus de collines" --provider deepseek

# Un SVG existant comme référence (hors-ligne)
python svg_fit.py --llm-svg-file dessin.svg --shapes 60
```

**Entraînement à partir du SVG du LLM + vérification de proximité.** Avec
`--from-llm-svg`, le LLM produit un document SVG (contraint aux primitives
simples `rect`/`circle`/`ellipse`/`polygon`/`line` pour rester rasterisable), on
le rasterise en image cible, on entraîne notre SVG dessus, puis on **vérifie que
l'image de notre SVG est proche de celle du SVG du LLM** — rapport `MSE` et
`PSNR` (dB) avec un verdict `PROCHE ✓`. Le rasteriseur utilise `cairosvg` s'il
est installé (support complet, `path` inclus), sinon un rasteriseur minimal via
PIL.

## Limites (honnêtes)

- Les images sont conditionnées sur des captions synthétiques construites à
  partir des classes CIFAR-10 : les prompts efficaces sont ceux qui
  ressemblent aux templates d'entraînement (« une photo de chat », « un
  camion », ...). Ce n'est pas un modèle text-to-image généraliste.
- Le générateur de code entraîné *from scratch* sur 30 exemples **mémorise**
  plutôt qu'il ne généralise. Pour un vrai assistant de code, partez d'un
  modèle pré-entraîné et faites du fine-tuning ; ce dépôt fournit la chaîne
  complète (données → entraînement → sampling) à petite échelle.
