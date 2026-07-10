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
