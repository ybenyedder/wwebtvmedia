"""
Génère dynamiquement des vecteurs d'entraînement (prompt, code) via un LLM,
au format JSONL consommé par main.py (data/code_pairs.jsonl).

Deux fournisseurs sont pris en charge (--provider) :
  - anthropic : API Claude (SDK `anthropic`, sorties structurées natives) ;
  - deepseek  : API DeepSeek, compatible OpenAI (SDK `openai`, mode JSON).

Deux phases, comme demandé — on fait générer le prompt PUIS le code au LLM :
  Phase 1  PROMPTS : le modèle produit des descriptions de tâches de code
                     variées (sortie structurée, dédupliquées par catégorie).
  Phase 2  CODE    : pour chaque prompt, le modèle écrit le code correspondant
                     (sortie structurée : un seul bloc de code par prompt).

Connexion dynamique à main.py : le script écrit dans le MÊME fichier que
main.py lit (CODE_DATA), respecte sa contrainte de longueur (MAX_CODE_LEN) et
valide le résultat final avec son propre chargeur CodePairDataset.

Auth (clés d'API), résolues dans l'environnement :
  - anthropic : ANTHROPIC_API_KEY (ou profil `ant auth login`) ;
  - deepseek  : DEEPSEEK_API_KEY.
Passez --dry-run pour tester toute la chaîne hors-ligne, sans clé ni API.

Exemples :
  export ANTHROPIC_API_KEY=sk-ant-...
  python generate_dataset.py --num 300                       # Claude (défaut)
  export DEEPSEEK_API_KEY=sk-...
  python generate_dataset.py --provider deepseek --num 300   # DeepSeek
  python generate_dataset.py --dry-run --num 20              # hors-ligne, sans API
  python main.py train-code --data data/code_pairs.jsonl     # entraîne dessus
"""

import argparse
import json
import os
import sys

from pydantic import BaseModel, Field

# --- Connexion dynamique à main.py (sans forcer l'import de torch) ---
DEFAULT_OUTPUT = "data/code_pairs.jsonl"
MAX_CODE_LEN = 512          # valeur de repli si main.py n'est pas importable
try:
    import main as _pipeline
    DEFAULT_OUTPUT = _pipeline.CODE_DATA
    MAX_CODE_LEN = _pipeline.MAX_CODE_LEN
except Exception as exc:  # torch/torchvision absents : on reste autonome
    _pipeline = None
    print(f"[i] main.py non importé ({exc.__class__.__name__}) — valeurs par "
          f"défaut utilisées (sortie={DEFAULT_OUTPUT}, max_len={MAX_CODE_LEN}).")

PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-opus-4-8",
    "deepseek": "deepseek-chat",
}
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Catégories parcourues en boucle pour forcer la diversité des prompts.
CATEGORIES = [
    "manipulation de chaînes de caractères",
    "listes, ensembles et dictionnaires",
    "algorithmes classiques (tri, recherche, parcours)",
    "mathématiques et arithmétique",
    "lecture/écriture de fichiers et formats (JSON, CSV)",
    "programmation orientée objet (classes, méthodes)",
    "décorateurs, générateurs et fonctions d'ordre supérieur",
    "dates, heures et durées",
    "récursivité et backtracking",
    "traitement de données et agrégations",
    "expressions régulières et validation",
    "gestion des erreurs et cas limites",
]


# --- Schémas de sortie structurée (garantissent un JSON valide) ---
class PromptList(BaseModel):
    prompts: list[str] = Field(description="Descriptions de tâches de code, "
                                           "une par entrée, en français.")


class CodeSolution(BaseModel):
    code: str = Field(description="Le code source seul, sans texte ni ``` autour.")


def _parse_lenient(text, schema):
    """Valide `text` contre `schema`, en tolérant les défauts fréquents des
    LLM : balises ``` autour du JSON et sauts de ligne littéraux dans les
    chaînes (json.loads(strict=False) les accepte)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]           # retire ```json ... ```
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return schema.model_validate_json(text)
    except Exception:
        return schema.model_validate(json.loads(text, strict=False))


def _strict_schema(schema):
    """Ajoute additionalProperties=false à chaque objet (exigé par les
    sorties structurées de type json_schema)."""
    node = schema.model_json_schema()

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object":
                n["additionalProperties"] = False
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(node)
    return node


# --- Fournisseurs ---
# Chaque fournisseur expose .structured(system, user, schema, max_tokens) et
# retourne une instance validée du schéma Pydantic demandé.
class AnthropicProvider:
    def __init__(self, model):
        try:
            import anthropic
        except ImportError:
            sys.exit("Le paquet 'anthropic' est requis : pip install anthropic")
        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            sys.exit(
                "Aucune clé d'API Anthropic trouvée. Définissez ANTHROPIC_API_KEY "
                "(export ANTHROPIC_API_KEY=sk-ant-...) ou connectez-vous via "
                "`ant auth login`. Pour tester sans API : --dry-run.")
        self.model = model
        self.client = anthropic.Anthropic(max_retries=4)

    def structured(self, system, user, schema, max_tokens):
        try:
            resp = self.client.messages.parse(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
            return resp.parsed_output
        except AttributeError:
            # SDK plus ancien sans messages.parse : output_config.format
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {
                    "type": "json_schema", "schema": _strict_schema(schema)}},
            )
            text = next(b.text for b in resp.content if b.type == "text")
            return _parse_lenient(text, schema)


class DeepSeekProvider:
    """API DeepSeek, compatible OpenAI. Utilise le mode JSON (response_format
    json_object) + le schéma injecté dans le prompt, puis valide avec Pydantic.
    DeepSeek exige que le mot « json » figure dans les messages : le schéma
    injecté le garantit."""

    def __init__(self, model):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("Le paquet 'openai' est requis pour DeepSeek : "
                     "pip install openai")
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            sys.exit(
                "Aucune clé d'API DeepSeek trouvée. Définissez DEEPSEEK_API_KEY "
                "(export DEEPSEEK_API_KEY=sk-...). Pour tester sans API : "
                "--dry-run.")
        self.model = model
        self.client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL,
                             max_retries=4)

    def structured(self, system, user, schema, max_tokens):
        fmt = json.dumps(_strict_schema(schema), ensure_ascii=False)
        sys_msg = (system + "\n\nRéponds UNIQUEMENT par un objet JSON valide, "
                   "sans texte autour, respectant strictement ce schéma :\n"
                   + fmt)
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user}],
        )
        return _parse_lenient(resp.choices[0].message.content, schema)


def make_provider(name, model):
    if name == "anthropic":
        return AnthropicProvider(model)
    if name == "deepseek":
        return DeepSeekProvider(model)
    sys.exit(f"Fournisseur inconnu : {name}")


# --- Phase 1 : génération des prompts ---
def generate_prompts(provider, language, category, count, avoid):
    system = (
        "Tu génères des énoncés d'exercices de programmation courts, clairs et "
        "variés, destinés à entraîner un modèle de génération de code. Chaque "
        "énoncé décrit UNE tâche réalisable en une fonction ou une petite classe."
    )
    avoid_block = ""
    if avoid:
        sample = "\n".join(f"- {p}" for p in list(avoid)[-40:])
        avoid_block = ("\n\nÉvite de répéter ou de reformuler ces énoncés déjà "
                       f"générés :\n{sample}")
    user = (
        f"Langage cible : {language}.\n"
        f"Thème : {category}.\n"
        f"Génère {count} énoncés DISTINCTS sur ce thème, en français, "
        f"commençant par un verbe à l'impératif (« écris une fonction qui... »). "
        f"Chaque énoncé doit être court (une phrase) et sans exemple de code."
        f"{avoid_block}"
    )
    result = provider.structured(system, user, PromptList, max_tokens=2000)
    return [p.strip() for p in result.prompts if p.strip()]


# --- Phase 2 : génération du code pour un prompt ---
def generate_code(provider, language, prompt):
    system = (
        f"Tu es un expert {language}. Tu écris du code correct, idiomatique et "
        f"CONCIS (idéalement moins de 30 lignes). Tu réponds uniquement par le "
        f"code source, sans explication, sans texte, sans balises Markdown."
    )
    user = f"Écris le code {language} pour la tâche suivante :\n{prompt}"
    result = provider.structured(system, user, CodeSolution, max_tokens=1200)
    return result.code.strip()


# --- Générateur factice pour --dry-run (aucune API) ---
def dry_run_prompts(language, category, count, seed):
    return [f"écris une fonction {language.lower()} pour {category} "
            f"(variante {seed + i})" for i in range(count)]


def dry_run_code(prompt, seed):
    return (f"def tache_{seed}(x):\n"
            f"    # {prompt[:50]}\n"
            f"    return x")


# --- Utilitaires I/O ---
def load_existing_prompts(path):
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["prompt"].strip().lower())
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def within_budget(prompt, code, max_bytes):
    # main.py tronque la séquence [BOS] prompt [SEP] code [EOS] à MAX_CODE_LEN
    # tokens (byte-level) : on garde les paires qui tiennent réellement.
    return len(prompt.encode("utf-8")) + len(code.encode("utf-8")) <= max_bytes


# --- Boucle principale ---
def run(args):
    model = args.model or PROVIDER_DEFAULT_MODEL[args.provider]
    max_bytes = args.max_bytes
    seen = set()
    mode = "a" if args.append else "w"
    if args.append:
        seen = load_existing_prompts(args.output)
        print(f"[i] {len(seen)} prompt(s) déjà présent(s) dans '{args.output}' "
              f"(déduplication active).")
    elif os.path.exists(args.output):
        print(f"[!] '{args.output}' sera écrasé (--overwrite).")

    provider = None
    if not args.dry_run:
        provider = make_provider(args.provider, model)
        print(f"[i] Fournisseur : {args.provider} | modèle : {model}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    kept, skipped_len, skipped_dup, failed, cat_idx = 0, 0, 0, 0, 0
    with open(args.output, mode, encoding="utf-8") as out:
        while kept < args.num:
            category = CATEGORIES[cat_idx % len(CATEGORIES)]
            cat_idx += 1
            need = min(args.batch_size, args.num - kept)

            # Phase 1 : prompts
            try:
                if args.dry_run:
                    prompts = dry_run_prompts(args.language, category, need, kept)
                else:
                    prompts = generate_prompts(provider, args.language,
                                               category, need, seen)
            except Exception as exc:
                print(f"[!] Phase 1 échouée ({category}) : {exc}. On continue.")
                continue

            # Phase 2 : code pour chaque prompt
            for prompt in prompts:
                if kept >= args.num:
                    break
                key = prompt.strip().lower()
                if key in seen:
                    skipped_dup += 1
                    continue
                try:
                    if args.dry_run:
                        code = dry_run_code(prompt, kept)
                    else:
                        code = generate_code(provider, args.language, prompt)
                except Exception as exc:
                    failed += 1
                    print(f"[!] Code échoué pour « {prompt[:50]}... » : {exc}")
                    continue

                if not code or not within_budget(prompt, code, max_bytes):
                    skipped_len += 1
                    continue

                out.write(json.dumps({"prompt": prompt, "code": code},
                                     ensure_ascii=False) + "\n")
                out.flush()  # écriture immédiate : un crash conserve le progrès
                seen.add(key)
                kept += 1
                if kept % 10 == 0 or kept == args.num:
                    print(f"  {kept}/{args.num} paires écrites "
                          f"(catégorie : {category})")

    print(f"\nTerminé : {kept} paires ajoutées à '{args.output}'.")
    if skipped_dup or skipped_len or failed:
        print(f"  ignorées — doublons : {skipped_dup}, trop longues "
              f"(> {max_bytes} octets) : {skipped_len}, échecs API : {failed}")

    # Validation via le vrai chargeur de main.py, si disponible
    if _pipeline is not None:
        try:
            ds = _pipeline.CodePairDataset(args.output)
            print(f"[✓] Validation main.CodePairDataset : {len(ds)} paires "
                  f"exploitables pour l'entraînement.")
        except SystemExit as exc:
            print(f"[!] Validation : {exc}")
    else:
        print("[i] Validation CodePairDataset ignorée (main.py non importable).")


def main():
    p = argparse.ArgumentParser(
        description="Génère des vecteurs (prompt, code) via un LLM (Claude ou "
                    "DeepSeek) pour entraîner le générateur de code de main.py.")
    p.add_argument("--provider", choices=list(PROVIDER_DEFAULT_MODEL),
                   default="anthropic",
                   help="Fournisseur LLM (défaut : anthropic).")
    p.add_argument("--num", type=int, default=200,
                   help="Nombre de paires à générer (défaut : 200).")
    p.add_argument("--batch-size", type=int, default=15,
                   help="Prompts demandés par appel en phase 1 (défaut : 15).")
    p.add_argument("--model", default=None,
                   help="Modèle à utiliser (défaut : selon le fournisseur — "
                        "claude-opus-4-8 / deepseek-chat).")
    p.add_argument("--language", default="Python",
                   help="Langage cible du code généré (défaut : Python).")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"Fichier JSONL de sortie (défaut : {DEFAULT_OUTPUT}).")
    p.add_argument("--max-bytes", type=int, default=MAX_CODE_LEN,
                   help="Budget d'octets prompt+code au-delà duquel une paire "
                        f"est ignorée (défaut : MAX_CODE_LEN={MAX_CODE_LEN}).")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--append", dest="append", action="store_true",
                       default=True, help="Ajoute au fichier existant (défaut).")
    group.add_argument("--overwrite", dest="append", action="store_false",
                       help="Écrase le fichier de sortie.")
    p.add_argument("--dry-run", action="store_true",
                   help="Génère des paires factices hors-ligne (sans API).")
    run(p.parse_args())


if __name__ == "__main__":
    main()
