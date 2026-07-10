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
pip install -r requirements.txt
```

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
python main.py export-onnx
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
- Les latents sont **normalisés par canal** (statistiques stockées dans le
  checkpoint) avant le flow matching, pour partir d'un bruit N(0,1) cohérent.
- **EMA** (decay 0,999) sur le réseau de drift, utilisée pour l'export et la
  génération.
- Checkpoints à écriture **atomique**, reprise automatique phase/époque.
- Export ONNX (opset 18) vérifié par comparaison PyTorch/onnxruntime en
  batch > 1.

## Limites (honnêtes)

- Les images sont conditionnées sur des captions synthétiques construites à
  partir des classes CIFAR-10 : les prompts efficaces sont ceux qui
  ressemblent aux templates d'entraînement (« une photo de chat », « un
  camion », ...). Ce n'est pas un modèle text-to-image généraliste.
- Le générateur de code entraîné *from scratch* sur 30 exemples **mémorise**
  plutôt qu'il ne généralise. Pour un vrai assistant de code, partez d'un
  modèle pré-entraîné et faites du fine-tuning ; ce dépôt fournit la chaîne
  complète (données → entraînement → sampling) à petite échelle.
