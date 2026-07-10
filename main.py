"""
wwebtvmedia — deux générateurs pilotés par un prompt texte.

1. IMAGES : VAE conditionné par prompt + flow matching latent (CIFAR-10).
   Un encodeur de texte byte-level remplace l'embedding de classe ; les
   captions d'entraînement sont synthétisées à partir des 10 classes.
   L'export ONNX embarque l'encodeur de texte, l'intégration RK4 et le
   décodeur dans un seul graphe à batch dynamique.

2. CODE : transformer décodeur byte-level entraîné sur des paires
   (prompt, code) au format JSONL — voir data/code_pairs.jsonl.
   La perte n'est appliquée que sur la partie code (après le séparateur).

CLI :
  python main.py train-image   [--vae-epochs 15 --drift-epochs 30]
  python main.py train-code    [--data data/code_pairs.jsonl --epochs 100]
  python main.py generate-image --prompt "une photo de chat" --n 8
  python main.py generate-code  --prompt "écris une fonction fibonacci"
  python main.py export-onnx
"""

import argparse
import json
import math
import os

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if not IN_COLAB and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")  # environnement sans écran : on sauvegarde, sans afficher
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

# Tokenisation byte-level partagée (aucun vocabulaire externe à télécharger)
PAD, BOS, SEP, EOS = 0, 1, 2, 3
BYTE_OFFSET = 4
VOCAB_SIZE = 256 + BYTE_OFFSET
MAX_PROMPT_LEN = 64          # longueur (fixe) des prompts côté image
MAX_CODE_LEN = 512           # longueur max prompt+code côté code

# Générateur d'images
LATENT_CHANNELS = 4
LATENT_HW = 8                # 32x32 avec deux réductions de moitié -> 8x8
COND_DIM = 128
BATCH_SIZE = 128
VAE_EPOCHS = 15
DRIFT_EPOCHS = 30
LR_VAE = 1e-3
LR_DRIFT = 2e-4
BETA_KL = 1e-3
EMA_DECAY = 0.999
GRAD_CLIP = 1.0
ODE_STEPS = 20               # pas RK4 déroulés dans le graphe ONNX
IMG_CKPT = "image_checkpoint.pth"
ONNX_PATH = "wwebtvmedia_image_generator.onnx"
SAMPLES_PATH = "samples_grid.png"

# Générateur de code
CODE_DIM = 256
CODE_HEADS = 8
CODE_LAYERS = 4
CODE_EPOCHS = 100
CODE_BATCH = 16
LR_CODE = 3e-4
CODE_CKPT = "code_checkpoint.pth"
CODE_DATA = "data/code_pairs.jsonl"

NUM_WORKERS = min(4, os.cpu_count() or 2)

CIFAR10_CLASSES = ["avion", "automobile", "oiseau", "chat", "cerf",
                   "chien", "grenouille", "cheval", "bateau", "camion"]
CAPTION_TEMPLATES = ["une photo de {}", "une image de {}", "{}"]


# --- TOKENISATION BYTE-LEVEL ---
def text_to_ids(s):
    return [b + BYTE_OFFSET for b in s.encode("utf-8")]


def ids_to_text(ids):
    return bytes(i - BYTE_OFFSET for i in ids if i >= BYTE_OFFSET).decode(
        "utf-8", errors="replace")


def encode_prompt(s, max_len=MAX_PROMPT_LEN):
    """[BOS] texte [EOS], tronqué puis complété par PAD -> tenseur [max_len]."""
    ids = [BOS] + text_to_ids(s)[: max_len - 2] + [EOS]
    ids += [PAD] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# --- ENCODEUR DE TEXTE (conditionnement des images) ---
class TextEncoder(nn.Module):
    """Petit transformer encodeur byte-level -> vecteur de conditionnement."""

    def __init__(self, vocab=VOCAB_SIZE, dim=COND_DIM, max_len=MAX_PROMPT_LEN,
                 layers=2, heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, ids):
        pad_mask = ids == PAD
        h = self.tok(ids) + self.pos[:, : ids.size(1)]
        h = self.enc(h, src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp_min(1e-5)
        return self.out(pooled)


# --- BRIQUES IMAGE ---
class TimeEmbedding(nn.Module):
    """Embedding sinusoïdal du temps t ∈ [0,1], suivi d'un petit MLP."""

    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        args = t.float().view(-1, 1) * self.freqs.view(1, -1)
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))


class ResFiLMBlock(nn.Module):
    """Bloc résiduel pré-activation modulé par FiLM (projection zero-init)."""

    def __init__(self, channels, cond_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        scale, shift = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return x + h


class PromptConditionedVAE(nn.Module):
    """VAE image <-> latent, conditionné par un vecteur de prompt déjà encodé."""

    def __init__(self, cond_dim=COND_DIM, base=64):
        super().__init__()
        # Encodeur : 32 -> 16 -> 8
        self.enc_in = nn.Conv2d(3, base, 3, padding=1)
        self.enc_block1 = ResFiLMBlock(base, cond_dim)
        self.enc_down1 = nn.Conv2d(base, base, 4, stride=2, padding=1)
        self.enc_block2 = ResFiLMBlock(base, cond_dim)
        self.enc_down2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.enc_block3 = ResFiLMBlock(base * 2, cond_dim)
        self.enc_out = nn.Conv2d(base * 2, LATENT_CHANNELS * 2, 3, padding=1)
        # Décodeur : 8 -> 16 -> 32
        self.dec_in = nn.Conv2d(LATENT_CHANNELS, base * 2, 3, padding=1)
        self.dec_block1 = ResFiLMBlock(base * 2, cond_dim)
        self.dec_up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec_block2 = ResFiLMBlock(base, cond_dim)
        self.dec_up2 = nn.ConvTranspose2d(base, base, 4, stride=2, padding=1)
        self.dec_block3 = ResFiLMBlock(base, cond_dim)
        self.dec_out = nn.Conv2d(base, 3, 3, padding=1)

    def encode(self, x, cond):
        h = self.enc_in(x)
        h = self.enc_block1(h, cond)
        h = self.enc_down1(h)
        h = self.enc_block2(h, cond)
        h = self.enc_down2(h)
        h = self.enc_block3(h, cond)
        mu, logvar = self.enc_out(h).chunk(2, dim=1)
        return mu, logvar.clamp(-30, 20)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z, cond):
        h = self.dec_in(z)
        h = self.dec_block1(h, cond)
        h = self.dec_up1(h)
        h = self.dec_block2(h, cond)
        h = self.dec_up2(h)
        h = self.dec_block3(h, cond)
        return torch.tanh(self.dec_out(F.silu(h)))


class DriftNetwork(nn.Module):
    """Champ de vitesse v(z, t, cond) pour le flow matching latent."""

    def __init__(self, cond_dim=COND_DIM, width=128, depth=4):
        super().__init__()
        self.time_emb = TimeEmbedding(cond_dim)
        self.in_conv = nn.Conv2d(LATENT_CHANNELS, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ResFiLMBlock(width, cond_dim) for _ in range(depth)])
        self.out_norm = nn.GroupNorm(8, width)
        self.out_conv = nn.Conv2d(width, LATENT_CHANNELS, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z, t, cond):
        c = cond + self.time_emb(t)
        h = self.in_conv(z)
        for block in self.blocks:
            h = block(h, c)
        return self.out_conv(F.silu(self.out_norm(h)))


class ONNXImageGenerator(nn.Module):
    """Générateur exportable : encodeur de texte + RK4 déroulé + décodage.

    Entrées : latent_noise [N,4,8,8] et prompt_ids [N,64] (byte-level,
    voir encode_prompt). Le temps est dérivé de z pour que le batch reste
    réellement dynamique dans le graphe tracé.
    """

    def __init__(self, text_encoder, vae, drift, latent_mean, latent_std,
                 steps=ODE_STEPS):
        super().__init__()
        self.text_encoder = text_encoder
        self.vae = vae
        self.drift = drift
        self.steps = steps
        self.dt = 1.0 / steps
        self.register_buffer("latent_mean", latent_mean.view(1, -1, 1, 1).float())
        self.register_buffer("latent_std", latent_std.view(1, -1, 1, 1).float())

    def forward(self, z, prompt_ids):
        cond = self.text_encoder(prompt_ids)
        dt = self.dt
        for i in range(self.steps):
            t = torch.ones_like(z[:, 0, 0, 0]) * (i * dt)
            k1 = self.drift(z, t, cond)
            k2 = self.drift(z + k1 * (dt / 2), t + dt / 2, cond)
            k3 = self.drift(z + k2 * (dt / 2), t + dt / 2, cond)
            k4 = self.drift(z + k3 * dt, t + dt, cond)
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        z = z * self.latent_std + self.latent_mean
        img = self.vae.decode(z, cond)
        return torch.clamp((img + 1) / 2.0, 0.0, 1.0)


# --- PERTES IMAGE ---
def vae_loss(vae, imgs, cond, beta=BETA_KL):
    mu, logvar = vae.encode(imgs, cond)
    z = vae.reparameterize(mu, logvar)
    recon = vae.decode(z, cond)
    recon_loss = F.mse_loss(recon, imgs)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def flow_matching_loss(drift, z1, cond):
    """Rectified flow : interpolation linéaire bruit -> donnée, cible z1 - z0."""
    t = torch.rand(z1.size(0), device=z1.device)
    z0 = torch.randn_like(z1)
    t_ = t.view(-1, 1, 1, 1)
    zt = (1 - t_) * z0 + t_ * z1
    return F.mse_loss(drift(zt, t, cond), z1 - z0)


# --- OUTILS ---
class EMA:
    """Moyenne mobile exponentielle des poids (qualité d'échantillonnage)."""

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.detach().clone() for k, v in sd.items()}


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build_caption_bank():
    """Tenseur [10 classes, T templates, MAX_PROMPT_LEN] de captions tokenisées."""
    return torch.stack([
        torch.stack([encode_prompt(tpl.format(cls)) for tpl in CAPTION_TEMPLATES])
        for cls in CIFAR10_CLASSES
    ])


def captions_for(labels_cpu, bank):
    """Caption aléatoire (template au hasard) pour chaque label du batch."""
    tpl = torch.randint(0, bank.size(1), (labels_cpu.size(0),))
    return bank[labels_cpu, tpl]


@torch.no_grad()
def compute_latent_stats(vae, text_enc, loader, bank, n_batches=50):
    """Moyenne/écart-type par canal des latents, pour les ramener vers N(0,1)."""
    vae.eval()
    text_enc.eval()
    feats = []
    for i, (imgs, lbls) in enumerate(loader):
        if i >= n_batches:
            break
        cond = text_enc(bank[lbls, 0].to(DEVICE))
        mu, logvar = vae.encode(imgs.to(DEVICE), cond)
        feats.append(vae.reparameterize(mu, logvar).float().cpu())
    z = torch.cat(feats)
    return z.mean(dim=(0, 2, 3)), z.std(dim=(0, 2, 3)).clamp_min(1e-4)


# --- ENTRAÎNEMENT IMAGE ---
def train_image(vae_epochs=VAE_EPOCHS, drift_epochs=DRIFT_EPOCHS):
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    print(f"Appareil : {DEVICE} | AMP : {USE_AMP}")

    text_enc = TextEncoder().to(DEVICE)
    vae = PromptConditionedVAE().to(DEVICE)
    drift = DriftNetwork().to(DEVICE)
    opt_vae = optim.AdamW(
        list(vae.parameters()) + list(text_enc.parameters()), lr=LR_VAE)
    opt_drift = optim.AdamW(drift.parameters(), lr=LR_DRIFT)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    bank = build_caption_bank()

    start_phase, start_epoch = 1, 0
    latent_mean, latent_std = None, None

    if os.path.exists(IMG_CKPT):
        print(f"Checkpoint trouvé ('{IMG_CKPT}'), restauration...")
        ckpt = torch.load(IMG_CKPT, map_location=DEVICE, weights_only=True)
        text_enc.load_state_dict(ckpt["text_enc_state"])
        vae.load_state_dict(ckpt["vae_state"])
        drift.load_state_dict(ckpt["drift_state"])
        opt_vae.load_state_dict(ckpt["opt_vae"])
        opt_drift.load_state_dict(ckpt["opt_drift"])
        latent_mean = ckpt.get("latent_mean")
        latent_std = ckpt.get("latent_std")
        start_phase, start_epoch = ckpt["phase"], ckpt["epoch"] + 1
        if start_phase == 1 and start_epoch >= vae_epochs:
            start_phase, start_epoch = 2, 0
        print(f"Reprise : phase {start_phase}, époque {start_epoch + 1}")

    ema = EMA(drift)
    if os.path.exists(IMG_CKPT):
        ema_state = torch.load(IMG_CKPT, map_location=DEVICE,
                               weights_only=True).get("ema_state")
        if ema_state is not None:
            ema.load_state_dict(ema_state)

    loader = DataLoader(
        datasets.CIFAR10(
            root="./data", train=True, download=True,
            transform=T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)]),
        ),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
        persistent_workers=NUM_WORKERS > 0,
    )

    def save_checkpoint(phase, epoch):
        atomic_save({
            "phase": phase,
            "epoch": epoch,
            "text_enc_state": text_enc.state_dict(),
            "vae_state": vae.state_dict(),
            "drift_state": drift.state_dict(),
            "opt_vae": opt_vae.state_dict(),
            "opt_drift": opt_drift.state_dict(),
            "ema_state": ema.state_dict(),
            "latent_mean": latent_mean,
            "latent_std": latent_std,
        }, IMG_CKPT)

    # PHASE 1 : VAE + encodeur de texte, entraînés conjointement
    if start_phase == 1:
        print("\n--- PHASE 1 : VAE + encodeur de texte ---")
        for epoch in range(start_epoch, vae_epochs):
            vae.train()
            text_enc.train()
            pbar = tqdm(loader, desc=f"VAE {epoch + 1}/{vae_epochs}")
            for imgs, lbls in pbar:
                cap = captions_for(lbls, bank).to(DEVICE)
                imgs = imgs.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                    cond = text_enc(cap)
                    loss, recon_l, kl_l = vae_loss(vae, imgs, cond)
                opt_vae.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt_vae)
                nn.utils.clip_grad_norm_(
                    list(vae.parameters()) + list(text_enc.parameters()), GRAD_CLIP)
                scaler.step(opt_vae)
                scaler.update()
                pbar.set_postfix(recon=f"{recon_l.item():.4f}",
                                 kl=f"{kl_l.item():.3f}")
            save_checkpoint(phase=1, epoch=epoch)
        start_epoch = 0

    # Geler VAE et encodeur de texte
    for m in (vae, text_enc):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    if latent_mean is None:
        print("\nCalcul des statistiques latentes...")
        latent_mean, latent_std = compute_latent_stats(vae, text_enc, loader, bank)
        print(f"  mean={latent_mean.tolist()}\n  std ={latent_std.tolist()}")
        save_checkpoint(phase=2, epoch=-1)
    mean_dev = latent_mean.view(1, -1, 1, 1).to(DEVICE)
    std_dev = latent_std.view(1, -1, 1, 1).to(DEVICE)

    # PHASE 2 : flow matching dans l'espace latent normalisé
    if start_phase == 2:
        print("\n--- PHASE 2 : Drift Network (flow matching) ---")
        for epoch in range(start_epoch, drift_epochs):
            drift.train()
            pbar = tqdm(loader, desc=f"Flow {epoch + 1}/{drift_epochs}")
            for imgs, lbls in pbar:
                cap = captions_for(lbls, bank).to(DEVICE)
                imgs = imgs.to(DEVICE, non_blocking=True)
                with torch.no_grad():
                    cond = text_enc(cap)
                    mu, logvar = vae.encode(imgs, cond)
                    z1 = (vae.reparameterize(mu, logvar) - mean_dev) / std_dev
                with torch.amp.autocast(DEVICE, enabled=USE_AMP):
                    loss = flow_matching_loss(drift, z1, cond)
                opt_drift.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt_drift)
                nn.utils.clip_grad_norm_(drift.parameters(), GRAD_CLIP)
                scaler.step(opt_drift)
                scaler.update()
                ema.update(drift)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            save_checkpoint(phase=2, epoch=epoch)

    print("\nEntraînement image terminé.")
    export_onnx()
    demo_prompts = [f"une photo de {c}" for c in CIFAR10_CLASSES]
    generate_images(demo_prompts, n_per_prompt=8, out_path=SAMPLES_PATH)
    if IN_COLAB:
        print("Téléchargement des fichiers (grille, checkpoint, ONNX)...")
        files.download(SAMPLES_PATH)
        files.download(IMG_CKPT)
        files.download(ONNX_PATH)


def build_generator_from_checkpoint():
    if not os.path.exists(IMG_CKPT):
        raise SystemExit(
            f"'{IMG_CKPT}' introuvable : lancez d'abord `python main.py train-image`.")
    ckpt = torch.load(IMG_CKPT, map_location=DEVICE, weights_only=True)
    if ckpt.get("latent_mean") is None:
        raise SystemExit("Phase 1 incomplète : relancez `python main.py train-image`.")
    text_enc = TextEncoder().to(DEVICE)
    text_enc.load_state_dict(ckpt["text_enc_state"])
    vae = PromptConditionedVAE().to(DEVICE)
    vae.load_state_dict(ckpt["vae_state"])
    drift = DriftNetwork().to(DEVICE)
    drift.load_state_dict(ckpt.get("ema_state") or ckpt["drift_state"])
    gen = ONNXImageGenerator(text_enc, vae, drift,
                             ckpt["latent_mean"], ckpt["latent_std"]).to(DEVICE)
    return gen.eval()


def export_onnx():
    print("\n--- Exportation ONNX du générateur d'images ---")
    generator = build_generator_from_checkpoint()
    dummy_z = torch.randn(1, LATENT_CHANNELS, LATENT_HW, LATENT_HW, device=DEVICE)
    dummy_ids = encode_prompt("une photo de chat").unsqueeze(0).to(DEVICE)
    torch.onnx.export(
        generator, (dummy_z, dummy_ids), ONNX_PATH,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=["latent_noise", "prompt_ids"],
        output_names=["generated_image"],
        dynamic_axes={
            "latent_noise": {0: "batch_size"},
            "prompt_ids": {0: "batch_size"},
            "generated_image": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"Modèle ONNX sauvegardé : {ONNX_PATH}")

    # Vérification croisée PyTorch / onnxruntime, en batch > 1
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        z_t = torch.randn(2, LATENT_CHANNELS, LATENT_HW, LATENT_HW)
        ids_t = torch.stack([encode_prompt("une photo de chat"),
                             encode_prompt("une photo de camion")])
        ort_out = sess.run(None, {"latent_noise": z_t.numpy(),
                                  "prompt_ids": ids_t.numpy()})[0]
        with torch.no_grad():
            ref = generator(z_t.to(DEVICE), ids_t.to(DEVICE)).cpu().numpy()
        print(f"Vérification ONNX (batch=2) : écart max = "
              f"{np.abs(ort_out - ref).max():.2e}")
    except ImportError:
        print("onnxruntime absent : vérification ONNX ignorée.")


def generate_images(prompts, n_per_prompt=8, out_path=SAMPLES_PATH):
    generator = build_generator_from_checkpoint()
    all_prompts = [p for p in prompts for _ in range(n_per_prompt)]
    ids = torch.stack([encode_prompt(p) for p in all_prompts]).to(DEVICE)
    with torch.no_grad():
        z = torch.randn(len(all_prompts), LATENT_CHANNELS, LATENT_HW, LATENT_HW,
                        device=DEVICE)
        imgs = generator(z, ids)
    vutils.save_image(imgs, out_path, nrow=n_per_prompt)
    print(f"Grille sauvegardée : {out_path} (une ligne par prompt)")

    grid = vutils.make_grid(imgs.cpu(), nrow=n_per_prompt)
    plt.figure(figsize=(8, max(2, len(prompts))))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(" | ".join(prompts[:5]) + ("..." if len(prompts) > 5 else ""))
    plt.tight_layout()
    plt.show()


# --- GÉNÉRATEUR DE CODE ---
class CausalSelfAttention(nn.Module):
    """Attention multi-têtes causale avec cache clé/valeur optionnel.

    - `past_kv` : (k, v) accumulés des positions précédentes ; concaténés aux
      k/v courants pour un décodage incrémental.
    - `use_cache` : renvoie le nouveau (k, v) à réutiliser à l'étape suivante.
    - `attn_mask` : masque booléen (True = participe) pour le passage complet
      d'entraînement (causal + masque de padding sur les clés).
    """

    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v) if use_cache else None
        p = self.dropout if self.training else 0.0
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                 dropout_p=p)
        else:
            # Décodage incrémental (T=1) : la requête voit toutes les clés du
            # cache, ce qui est causal par construction — aucun masque requis.
            is_causal = past_kv is None and T > 1
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal,
                                                 dropout_p=p)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out), new_kv


class DecoderBlock(nn.Module):
    """Bloc décodeur pré-norm : attention causale + feed-forward."""

    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                nn.Linear(dim * 4, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False):
        a, new_kv = self.attn(self.norm1(x), attn_mask, past_kv, use_cache)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, new_kv


class CodeGenerator(nn.Module):
    """Transformer décodeur byte-level : prompt -> code (autorégressif).

    forward(ids)                       -> logits            (entraînement)
    forward(ids, use_cache=True)       -> logits, caches    (amorçage du cache)
    forward(ids, caches=..., use_cache=True) -> logits, caches  (pas incrémental)

    Le cache clé/valeur évite de recalculer l'attention sur tout le préfixe à
    chaque token généré : le décodage passe de O(n²) à O(n).
    """

    def __init__(self, vocab=VOCAB_SIZE, dim=CODE_DIM, heads=CODE_HEADS,
                 layers=CODE_LAYERS, max_len=MAX_CODE_LEN):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList(
            [DecoderBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids, caches=None, use_cache=False):
        B, T = ids.shape
        past_len = 0 if caches is None else caches[0][0].size(2)
        positions = torch.arange(past_len, past_len + T, device=ids.device)
        h = self.tok(ids) + self.pos[:, positions]

        attn_mask = None
        if caches is None and T > 1:
            # Passage complet : causal ET masque de padding sur les clés
            # (comme src_key_padding_mask) — les requêtes de padding ne sont
            # pas masquées, ce qui évite des lignes entièrement masquées (NaN).
            causal = torch.tril(
                torch.ones(T, T, dtype=torch.bool, device=ids.device))
            key_ok = (ids != PAD)[:, None, None, :]      # (B,1,1,T)
            attn_mask = causal[None, None] & key_ok      # (B,1,T,T)

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = None if caches is None else caches[i]
            h, kv = block(h, attn_mask=attn_mask, past_kv=past,
                          use_cache=use_cache)
            if use_cache:
                new_caches.append(kv)

        logits = self.head(self.norm(h))
        return (logits, new_caches) if use_cache else logits


class CodePairDataset(Dataset):
    """JSONL {"prompt": ..., "code": ...} -> séquences [BOS] prompt [SEP] code [EOS].

    La perte (labels) ne couvre que la partie code : le modèle apprend à
    générer du code, pas à recopier les prompts.
    """

    def __init__(self, path, max_len=MAX_CODE_LEN):
        self.samples = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                seq = ([BOS] + text_to_ids(item["prompt"]) + [SEP]
                       + text_to_ids(item["code"]) + [EOS])[:max_len]
                if SEP not in seq or seq.index(SEP) >= len(seq) - 1:
                    skipped += 1
                    continue
                ids = torch.tensor(seq, dtype=torch.long)
                labels = torch.full((len(seq),), -100, dtype=torch.long)
                sep_pos = seq.index(SEP)
                labels[sep_pos + 1:] = ids[sep_pos + 1:]
                self.samples.append((ids, labels))
        if skipped:
            print(f"[!] {skipped} exemple(s) ignoré(s) (prompt trop long).")
        if not self.samples:
            raise SystemExit(f"Aucun exemple exploitable dans '{path}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_code(batch):
    max_len = max(ids.size(0) for ids, _ in batch)
    ids_out = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    lbl_out = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        ids_out[i, : ids.size(0)] = ids
        lbl_out[i, : labels.size(0)] = labels
    return ids_out, lbl_out


def train_code(data_path=CODE_DATA, epochs=CODE_EPOCHS, batch_size=CODE_BATCH,
               lr=LR_CODE):
    torch.manual_seed(SEED)
    print(f"Appareil : {DEVICE}")
    dataset = CodePairDataset(data_path)
    print(f"{len(dataset)} paires (prompt, code) chargées depuis '{data_path}'.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_code)

    model = CodeGenerator().to(DEVICE)
    start_epoch = 0
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if os.path.exists(CODE_CKPT):
        print(f"Checkpoint trouvé ('{CODE_CKPT}'), restauration...")
        ckpt = torch.load(CODE_CKPT, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1

    model.train()
    for epoch in range(start_epoch, epochs):
        total, count = 0.0, 0
        pbar = tqdm(loader, desc=f"Code {epoch + 1}/{epochs}")
        for ids, labels in pbar:
            ids, labels = ids.to(DEVICE), labels.to(DEVICE)
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB_SIZE),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += loss.item()
            count += 1
            pbar.set_postfix(loss=f"{total / count:.4f}")
        atomic_save({"model_state": model.state_dict(),
                     "opt_state": opt.state_dict(),
                     "epoch": epoch}, CODE_CKPT)
    print(f"Entraînement code terminé, checkpoint : {CODE_CKPT}")
    if IN_COLAB:
        files.download(CODE_CKPT)


@torch.no_grad()
def sample_code(model, prompt, max_new=400, temperature=0.8, top_k=40):
    """Décodage autorégressif avec cache KV : le préfixe (prompt) n'est encodé
    qu'une fois, puis chaque token n'attend que les clés/valeurs en cache."""
    model.eval()
    ids = [BOS] + text_to_ids(prompt)[: MAX_CODE_LEN // 2] + [SEP]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    # Amorçage : un seul passage sur tout le prompt, on remplit le cache.
    logits, caches = model(x, use_cache=True)
    generated = []
    for _ in range(max_new):
        if len(ids) + len(generated) >= model.max_len:
            break
        step = logits[:, -1] / max(temperature, 1e-5)      # (1, vocab)
        if top_k > 0:
            kth = torch.topk(step, min(top_k, step.size(-1))).values[..., -1:]
            step = step.masked_fill(step < kth, float("-inf"))
        next_id = torch.multinomial(F.softmax(step, dim=-1), 1)  # (1, 1)
        if next_id.item() == EOS:
            break
        generated.append(next_id.item())
        # Pas incrémental : on ne passe QUE le nouveau token + le cache.
        logits, caches = model(next_id, caches=caches, use_cache=True)
    return ids_to_text(generated)


def generate_code(prompt, temperature=0.8, top_k=40):
    if not os.path.exists(CODE_CKPT):
        raise SystemExit(
            f"'{CODE_CKPT}' introuvable : lancez d'abord `python main.py train-code`.")
    model = CodeGenerator().to(DEVICE)
    model.load_state_dict(torch.load(CODE_CKPT, map_location=DEVICE,
                                     weights_only=True)["model_state"])
    code = sample_code(model, prompt, temperature=temperature, top_k=top_k)
    print(f"# Prompt : {prompt}\n{'-' * 60}\n{code}")
    return code


# --- CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="wwebtvmedia : génération d'images et de code par prompt.")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("train-image", help="Entraîne VAE + flow matching (CIFAR-10)")
    p.add_argument("--vae-epochs", type=int, default=VAE_EPOCHS)
    p.add_argument("--drift-epochs", type=int, default=DRIFT_EPOCHS)

    p = sub.add_parser("train-code", help="Entraîne le générateur de code")
    p.add_argument("--data", default=CODE_DATA)
    p.add_argument("--epochs", type=int, default=CODE_EPOCHS)
    p.add_argument("--batch-size", type=int, default=CODE_BATCH)
    p.add_argument("--lr", type=float, default=LR_CODE)

    p = sub.add_parser("generate-image", help="Génère des images depuis un prompt")
    p.add_argument("--prompt", action="append", required=True,
                   help="Répétable pour plusieurs prompts")
    p.add_argument("--n", type=int, default=8, help="Images par prompt")
    p.add_argument("--out", default=SAMPLES_PATH)

    p = sub.add_parser("generate-code", help="Génère du code depuis un prompt")
    p.add_argument("--prompt", required=True)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)

    sub.add_parser("export-onnx", help="Exporte le générateur d'images en ONNX")

    args = parser.parse_args()
    if args.cmd == "train-image":
        train_image(args.vae_epochs, args.drift_epochs)
    elif args.cmd == "train-code":
        train_code(args.data, args.epochs, args.batch_size, args.lr)
    elif args.cmd == "generate-image":
        generate_images(args.prompt, n_per_prompt=args.n, out_path=args.out)
    elif args.cmd == "generate-code":
        generate_code(args.prompt, args.temperature, args.top_k)
    elif args.cmd == "export-onnx":
        export_onnx()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
