"""
Support SVG : entraîne une représentation SVG par distance avec une image.

Principe (vectorisation d'image par optimisation, style « differentiable
vector graphics », en PyTorch pur — sans diffvg) :

  1. Une image SVG est paramétrée par N primitives (ellipses colorées : centre,
     rayons, rotation, couleur, opacité) + une couleur de fond.
  2. Un RASTERISEUR DIFFÉRENTIABLE rend ces primitives en une image pixel
     (couverture douce + compositing source-over) — les gradients traversent
     donc le rendu.
  3. On ENTRAÎNE les primitives par descente de gradient pour MINIMISER LA
     DISTANCE (MSE) entre le rendu et une image CIBLE.
  4. On exporte le résultat en vrai fichier .svg (ouvrable dans un navigateur)
     + un aperçu PNG du rendu.

La cible peut être :
  - une image produite par le générateur du pipeline (--from-prompt "..."),
  - un fichier image quelconque (--target chemin.png),
  - à défaut, une cible de démonstration synthétique.

Exemples :
  python svg_fit.py --shapes 60 --steps 500                 # démo synthétique
  python svg_fit.py --target photo.png --shapes 80 --out out.svg
  python svg_fit.py --from-prompt "une photo de chat" --shapes 60
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- RASTERISEUR SVG DIFFÉRENTIABLE ---
class DiffSVG(nn.Module):
    """N ellipses colorées composées sur un fond, rendues de façon
    différentiable. Les paramètres bruts sont transformés vers des plages
    valides (sigmoid/softplus) à chaque rendu."""

    def __init__(self, n_shapes=50, size=96, sharpness=40.0, target_mean=None):
        super().__init__()
        self.H = self.W = size
        self.sharpness = sharpness

        ys = torch.linspace(0, 1, size)
        xs = torch.linspace(0, 1, size)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("gx", gx)              # (H, W)
        self.register_buffer("gy", gy)

        # Colonnes : 0 cx, 1 cy, 2 rx, 3 ry, 4 theta, 5-7 rgb, 8 alpha
        raw = torch.zeros(n_shapes, 9)
        raw[:, 0:2] = torch.randn(n_shapes, 2) * 0.5      # centres -> ~[0,1]
        raw[:, 2:4] = -1.5 + torch.randn(n_shapes, 2) * 0.2  # rayons -> ~0.2
        raw[:, 4] = torch.randn(n_shapes) * 0.1           # rotation
        raw[:, 5:8] = torch.randn(n_shapes, 3) * 0.5      # couleurs
        raw[:, 8] = 0.0                                    # alpha -> 0.5
        self.shapes = nn.Parameter(raw)

        bg = torch.zeros(3)
        if target_mean is not None:
            m = target_mean.clamp(1e-3, 1 - 1e-3)
            bg = torch.log(m / (1 - m))                   # sigmoid^-1(moyenne)
        self.bg = nn.Parameter(bg)

    def params(self):
        """Valeurs transformées (dans leurs plages valides)."""
        p = self.shapes
        cx = torch.sigmoid(p[:, 0])
        cy = torch.sigmoid(p[:, 1])
        rx = F.softplus(p[:, 2]) + 1e-3
        ry = F.softplus(p[:, 3]) + 1e-3
        theta = p[:, 4]
        rgb = torch.sigmoid(p[:, 5:8])
        alpha = torch.sigmoid(p[:, 8])
        bg = torch.sigmoid(self.bg)
        return cx, cy, rx, ry, theta, rgb, alpha, bg

    def render(self):
        cx, cy, rx, ry, theta, rgb, alpha, bg = self.params()
        dx = self.gx[None] - cx[:, None, None]            # (N, H, W)
        dy = self.gy[None] - cy[:, None, None]
        c = torch.cos(theta)[:, None, None]
        s = torch.sin(theta)[:, None, None]
        u = dx * c + dy * s
        v = -dx * s + dy * c
        d = (u / rx[:, None, None]) ** 2 + (v / ry[:, None, None]) ** 2
        cov = torch.sigmoid((1.0 - d) * self.sharpness)   # (N, H, W) couverture
        a = cov * alpha[:, None, None]                    # opacité effective

        img = bg[:, None, None].expand(3, self.H, self.W).clone()
        for i in range(a.size(0)):                        # compositing arrière->avant
            ai = a[i][None]                               # (1, H, W)
            img = rgb[i][:, None, None] * ai + img * (1 - ai)
        return img.clamp(0, 1)                            # (3, H, W)

    def to_svg(self, view=100):
        """Exporte en balisage SVG valide (coordonnées mises à l'échelle)."""
        cx, cy, rx, ry, theta, rgb, alpha, bg = (t.detach() for t in self.params())
        br, bgc, bb = (int(round(float(x) * 255)) for x in bg)
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view} '
               f'{view}" width="{view}" height="{view}">',
               f'  <rect width="{view}" height="{view}" '
               f'fill="rgb({br},{bgc},{bb})"/>']
        for i in range(cx.size(0)):
            r, g, b = (int(round(float(x) * 255)) for x in rgb[i])
            deg = float(theta[i]) * 180.0 / math.pi
            CX, CY = float(cx[i]) * view, float(cy[i]) * view
            RX, RY = float(rx[i]) * view, float(ry[i]) * view
            out.append(
                f'  <g transform="rotate({deg:.2f} {CX:.2f} {CY:.2f})">'
                f'<ellipse cx="{CX:.2f}" cy="{CY:.2f}" rx="{RX:.2f}" '
                f'ry="{RY:.2f}" fill="rgb({r},{g},{b})" '
                f'fill-opacity="{float(alpha[i]):.3f}"/></g>')
        out.append("</svg>")
        return "\n".join(out)


# --- CIBLES ---
def load_image_target(path, size):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size))
    t = torch.from_numpy(_np_asarray(img)).float() / 255.0
    return t.permute(2, 0, 1)                              # (3, H, W)


def _np_asarray(img):
    import numpy as np
    return np.array(img)                     # copie (tenseur inscriptible)


# --- SVG DEPUIS UN LLM + RASTERISATION ---
_NAMED = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0),
          "green": (0, 128, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
          "orange": (255, 165, 0), "purple": (128, 0, 128), "gray": (128, 128, 128),
          "grey": (128, 128, 128), "none": None}


def _parse_color(s):
    if not s:
        return (0, 0, 0)
    s = s.strip().lower()
    if s in _NAMED:
        return _NAMED[s]
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    if s.startswith("rgb"):
        import re
        v = [int(float(x)) for x in re.findall(r"[\d.]+", s)[:3]]
        return tuple(v) if len(v) == 3 else (0, 0, 0)
    return (0, 0, 0)


def _rasterize_pil(svg_str, size):
    """Rasteriseur SVG minimal (rect/circle/ellipse/polygon/polyline/line)
    via PIL — sans dépendance système. Les transforms/paths sont ignorés."""
    import re
    import xml.etree.ElementTree as ET
    from PIL import Image, ImageDraw

    root = ET.fromstring(svg_str)
    vb = root.get("viewBox")
    if vb:
        _, _, vw, vh = [float(x) for x in re.split(r"[ ,]+", vb.strip())]
    else:
        vw = float(root.get("width", 100))
        vh = float(root.get("height", 100))
    sx, sy = size / vw, size / vh

    base = Image.new("RGBA", (size, size), (255, 255, 255, 255))

    def blit(draw_fn, color, opacity):
        if color is None:
            return
        overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        draw_fn(d, color + (int(round(opacity * 255)),))
        return Image.alpha_composite(base, overlay)

    for el in root.iter():
        tag = el.tag.split("}")[-1]
        fill = _parse_color(el.get("fill", "black"))
        op = float(el.get("fill-opacity", el.get("opacity", 1.0)))
        g = el.get
        if tag == "rect":
            x, y = float(g("x", 0)) * sx, float(g("y", 0)) * sy
            w, h = float(g("width", 0)) * sx, float(g("height", 0)) * sy
            base = blit(lambda d, c: d.rectangle([x, y, x + w, y + h], fill=c),
                        fill, op) or base
        elif tag == "circle":
            cx, cy, r = float(g("cx", 0)), float(g("cy", 0)), float(g("r", 0))
            bb = [(cx - r) * sx, (cy - r) * sy, (cx + r) * sx, (cy + r) * sy]
            base = blit(lambda d, c: d.ellipse(bb, fill=c), fill, op) or base
        elif tag == "ellipse":
            cx, cy = float(g("cx", 0)), float(g("cy", 0))
            rx, ry = float(g("rx", 0)), float(g("ry", 0))
            bb = [(cx - rx) * sx, (cy - ry) * sy, (cx + rx) * sx, (cy + ry) * sy]
            base = blit(lambda d, c: d.ellipse(bb, fill=c), fill, op) or base
        elif tag in ("polygon", "polyline"):
            nums = [float(v) for v in re.split(r"[ ,]+", g("points", "").strip())
                    if v]
            pts = [(nums[i] * sx, nums[i + 1] * sy)
                   for i in range(0, len(nums) - 1, 2)]
            if len(pts) >= 2:
                base = blit(lambda d, c: d.polygon(pts, fill=c), fill, op) or base
        elif tag == "line":
            stroke = _parse_color(g("stroke", "black"))
            pts = [(float(g("x1", 0)) * sx, float(g("y1", 0)) * sy),
                   (float(g("x2", 0)) * sx, float(g("y2", 0)) * sy)]
            w = max(1, int(float(g("stroke-width", 1)) * sx))
            base = blit(lambda d, c: d.line(pts, fill=c, width=w),
                        stroke, op) or base

    arr = torch.from_numpy(_np_asarray(base.convert("RGB"))).float() / 255.0
    return arr.permute(2, 0, 1)                            # (3, H, W)


def rasterize_svg(svg_str, size):
    """SVG -> image tensor (3,H,W). Utilise cairosvg si présent (support
    complet, paths inclus), sinon le rasteriseur PIL minimal."""
    try:
        import io
        import cairosvg
        from PIL import Image
        png = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=size,
                               output_height=size, background_color="white")
        img = Image.open(io.BytesIO(png)).convert("RGB")
        return torch.from_numpy(_np_asarray(img)).float().permute(2, 0, 1) / 255.0
    except Exception:
        return _rasterize_pil(svg_str, size)


def svg_from_llm(prompt, provider_name="anthropic", model=None):
    """Demande un document SVG à un LLM (Claude ou DeepSeek) via l'abstraction
    de fournisseur de generate_dataset.py. Contraint aux primitives simples
    pour rester rasterisable."""
    import generate_dataset as gd
    from pydantic import BaseModel, Field

    class SVGOut(BaseModel):
        svg: str = Field(description="Un document SVG complet et valide.")

    model = model or gd.PROVIDER_DEFAULT_MODEL[provider_name]
    provider = gd.make_provider(provider_name, model)
    system = (
        "Tu produis des illustrations SVG simples et VALIDES. Utilise "
        "UNIQUEMENT les balises <rect>, <circle>, <ellipse>, <polygon>, <line> "
        "avec des couleurs au format rgb(...) ou #hex et l'attribut "
        "fill-opacity. La racine doit avoir viewBox=\"0 0 100 100\". "
        "N'utilise NI <path>, NI <text>, NI transform, NI dégradé, NI clip."
    )
    user = f"Dessine en SVG : {prompt}. Réponds par le document SVG complet."
    return provider.structured(system, user, SVGOut, max_tokens=1500).svg.strip()


def compare(render, target):
    """Distance rendu vs cible : MSE + PSNR (dB)."""
    mse = F.mse_loss(render, target).item()
    psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
    return mse, psnr


def demo_target(size):
    """Cible synthétique : disque rouge + bande, sur fond turquoise."""
    ys = torch.linspace(0, 1, size)
    xs = torch.linspace(0, 1, size)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    img = torch.zeros(3, size, size)
    img[0], img[1], img[2] = 0.10, 0.55, 0.55             # fond turquoise
    disc = ((gx - 0.4) ** 2 + (gy - 0.45) ** 2) < 0.05
    for ch, val in enumerate((0.90, 0.20, 0.20)):         # disque rouge
        img[ch][disc] = val
    band = (gx > 0.6) & (gx < 0.85) & (gy > 0.2) & (gy < 0.8)
    for ch, val in enumerate((0.95, 0.85, 0.20)):         # bande jaune
        img[ch][band] = val
    return img


def prompt_target(prompt, size):
    """Image produite par le générateur du pipeline (nécessite le checkpoint
    image), puis redimensionnée — c'est « l'image générée » à vectoriser."""
    import main
    gen = main.build_generator_from_checkpoint()
    ids = main.encode_prompt(prompt).unsqueeze(0).to(main.DEVICE)
    z = torch.randn(1, main.LATENT_CHANNELS, main.LATENT_HW, main.LATENT_HW,
                    device=main.DEVICE)
    with torch.no_grad():
        img = gen(z, ids)                                 # (1, 3, 32, 32) in [0,1]
    img = F.interpolate(img, size=(size, size), mode="bilinear",
                        align_corners=False)
    return img[0].cpu()


# --- ENTRAÎNEMENT (distance avec l'image) ---
def fit(target, n_shapes=50, steps=500, lr=0.05, seed=0, log=True):
    torch.manual_seed(seed)
    size = target.size(-1)
    model = DiffSVG(n_shapes, size, target_mean=target.mean(dim=(1, 2)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    first, last = None, None
    for step in range(steps):
        render = model.render()
        loss = F.mse_loss(render, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
        if log and (step % max(1, steps // 10) == 0 or step == steps - 1):
            print(f"  step {step + 1}/{steps}  distance MSE = {last:.5f}")
    if log:
        print(f"Distance : {first:.5f} -> {last:.5f}")
    return model, last


def save_png(tensor, path):
    from PIL import Image
    import numpy as np
    arr = (tensor.detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255)
    Image.fromarray(arr.astype("uint8")).save(path)


def run(args):
    base = os.path.splitext(args.out)[0]
    llm_svg = None                          # SVG de référence (si source = SVG)

    if args.from_llm_svg:
        print(f"SVG généré par {args.provider} pour « {args.from_llm_svg} »")
        llm_svg = svg_from_llm(args.from_llm_svg, args.provider, args.model)
        with open(base + ".llm.svg", "w", encoding="utf-8") as f:
            f.write(llm_svg)
        print(f"SVG du LLM sauvegardé : {base}.llm.svg")
        target = rasterize_svg(llm_svg, args.size)
    elif args.llm_svg_file:
        print(f"SVG du LLM (fichier) : {args.llm_svg_file}")
        llm_svg = open(args.llm_svg_file, encoding="utf-8").read()
        target = rasterize_svg(llm_svg, args.size)
    elif args.from_prompt:
        print(f"Cible : image générée pour « {args.from_prompt} »")
        target = prompt_target(args.from_prompt, args.size)
    elif args.target:
        print(f"Cible : fichier {args.target}")
        target = load_image_target(args.target, args.size)
    else:
        print("Cible : image de démonstration synthétique")
        target = demo_target(args.size)

    print(f"Entraînement SVG (distance à l'image) : {args.shapes} ellipses, "
          f"{args.steps} étapes")
    model, final = fit(target, args.shapes, args.steps, args.lr, args.seed)

    svg = model.to_svg()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"SVG appris sauvegardé : {args.out}  ({args.shapes} ellipses)")

    render = model.render()
    save_png(render, base + "_render.png")
    save_png(target, base + "_target.png")
    print(f"Aperçus PNG : {base}_render.png (rendu) et {base}_target.png (cible)")

    # Vérification : l'image de NOTRE SVG est-elle proche de l'image cible ?
    mse, psnr = compare(render, target)
    label = "SVG du LLM" if llm_svg is not None else "cible"
    verdict = "PROCHE ✓" if psnr >= 20 else "écart notable"
    print(f"\nProximité (rendu de notre SVG vs {label}) : "
          f"MSE = {mse:.5f}, PSNR = {psnr:.1f} dB → {verdict}")


def main():
    p = argparse.ArgumentParser(
        description="Vectorise une image en SVG par rendu différentiable et "
                    "distance (MSE) à l'image.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--target", help="Image cible à vectoriser (fichier).")
    src.add_argument("--from-prompt",
                     help="Génère une image avec le pipeline puis la vectorise.")
    src.add_argument("--from-llm-svg",
                     help="Fait générer un SVG par le LLM (Claude/DeepSeek), le "
                          "rasterise et entraîne dessus.")
    src.add_argument("--llm-svg-file",
                     help="Utilise un fichier SVG existant comme référence LLM.")
    p.add_argument("--provider", choices=["anthropic", "deepseek"],
                   default="anthropic", help="Fournisseur LLM pour --from-llm-svg.")
    p.add_argument("--model", default=None, help="Modèle LLM (selon fournisseur).")
    p.add_argument("--shapes", type=int, default=50, help="Nombre d'ellipses.")
    p.add_argument("--steps", type=int, default=500, help="Étapes d'optimisation.")
    p.add_argument("--size", type=int, default=96, help="Résolution du rendu.")
    p.add_argument("--lr", type=float, default=0.05, help="Pas d'apprentissage.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="svg_output.svg", help="Fichier SVG de sortie.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
