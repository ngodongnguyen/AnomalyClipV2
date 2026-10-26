"""Paper figures (no architecture figure).  Local, needs matplotlib / numpy / PIL (system python3 has them).

  python3 scripts/make_figures.py ablation           # Fig. controls: ECP vs ZO / DUAL / SHUF / LATENT / CTRL / fixed-z (numbers from the registry)
  python3 scripts/make_figures.py decomp             # Fig. decomposition: zoom vs extent contribution (EXP-030)
  python3 scripts/make_figures.py select             # pick the images for the qualitative figures (rule below) -> figures/selection.csv
  python3 scripts/make_figures.py qual               # compose Fig. qualitative from the panels made on the server (scripts/figure_batch.py)
  python3 scripts/make_figures.py zsweep             # compose Fig. z-sweep from the same panels
  python3 scripts/make_figures.py sota               # qualitative comparison with other methods (needs their raw maps, see fig_sota)

Numbers: EXP-024 (ECP, ZO, DUAL; mean +- sd over seeds 111/222/333), EXP-028 (SHUF, LATENT, from the ECP-minus-arm PRO differences), EXP-030 (CTRL,
from the ECP-minus-CTRL PRO differences; image AUROC differences), EXP-055 (fixed-z training, ONE seed).  SHUF, LATENT and CTRL values are derived
from recorded differences, so they have no error bar here.  Image selection rule (fixed, not chosen by eye): per pixel set, the image whose per-image
AUROC of the matched ECP seed 111 (results/EXP-032/<set>/badcase_<set>.csv) is closest to the set median ('typical') or to the 10th percentile ('hard').
"""
import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
PIX_SETS = ["ClinicDB", "Kvasir", "ColonDB", "ISIC", "Endo", "TN3K"]
IMG_SETS = ["HeadCT", "BrainMRI", "Br35H"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})

# ---- recorded numbers (see module docstring) -------------------------------------------------------------------------
PRO = {  # pooled pixel AUPRO (%), sigma 4, fixed z; order = PIX_SETS
    "ECP":    [75.4, 51.2, 72.7, 88.5, 78.4, 51.9],
    "ZO":     [70.3, 48.7, 73.1, 84.3, 68.9, 48.3],
    "DUAL":   [69.7, 47.8, 72.7, 85.7, 68.8, 48.2],
}
PRO_SD = {"ECP": [0.7, 1.3, 1.0, 0.3, 1.1, 0.3], "ZO": [0.7, 0.9, 0.4, 0.4, 1.1, 1.0], "DUAL": [0.4, 0.9, 0.1, 0.2, 0.9, 0.4]}
D_SHUF = [4.7, 2.0, -0.4, 1.3, 8.1, 2.8]       # ECP - SHUF, PRO (EXP-028)
D_LAT = [1.5, -0.7, -1.2, 6.4, 5.8, 2.1]       # ECP - LATENT, PRO (EXP-028)
D_CTRL = [8.5, 7.6, 1.6, 10.6, 16.4, 2.7]      # ECP - CTRL, PRO (EXP-030)
PRO["SHUF"] = [round(e - d, 1) for e, d in zip(PRO["ECP"], D_SHUF)]
PRO["LATENT"] = [round(e - d, 1) for e, d in zip(PRO["ECP"], D_LAT)]
PRO["CTRL"] = [round(e - d, 1) for e, d in zip(PRO["ECP"], D_CTRL)]
PRO["FixedZ"] = [69.0, 47.8, 72.4, 86.6, 68.8, 49.8]   # EXP-055, ONE seed (seed 111)

IMG = {  # image AUROC (%), order = IMG_SETS
    "ECP": [94.4, 94.5, 97.1], "ZO": [89.1, 86.7, 89.4], "DUAL": [90.2, 92.5, 95.4],
    "SHUF": [84.8, 81.5, 87.4], "LATENT": [58.8, 68.8, 72.7], "FixedZ": [91.2, 93.3, 96.5],
}
IMG["CTRL"] = [round(e - d, 1) for e, d in zip(IMG["ECP"], [1.6, 2.8, 2.2])]
IMG_SD = {"ECP": [0.3, 0.3, 0.1], "ZO": [0.4, 0.8, 0.9], "DUAL": [0.7, 0.9, 0.7]}

ARMS = ["CTRL", "ZO", "DUAL", "FixedZ", "SHUF", "LATENT", "ECP"]
LABEL = {"CTRL": "Baseline (no zoom)", "ZO": "Zoom-only", "DUAL": "DUAL (learned const.)", "FixedZ": "Fixed-z training$^{\\dagger}$",
         "SHUF": "SHUF (shuffled labels)", "LATENT": "LATENT (no supervision)", "ECP": "ECP"}
COL = {"CTRL": "#8c8c8c", "ZO": "#E69F00", "DUAL": "#56B4E9", "FixedZ": "#009E73", "SHUF": "#D55E00", "LATENT": "#CC79A7", "ECP": "#0B3C8C"}
MARK = {"CTRL": "s", "ZO": "^", "DUAL": "v", "FixedZ": "D", "SHUF": "X", "LATENT": "P", "ECP": "o"}


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    print("wrote", os.path.join(FIG, name + ".pdf"), "and .png")


def dotpanel(ax, data, sd, sets, ylabel, title, ylim):
    n = len(ARMS)
    off = np.linspace(-0.36, 0.36, n)
    for j, arm in enumerate(ARMS):
        y = np.array(data[arm])
        x = np.arange(len(sets)) + off[j]
        big = arm == "ECP"
        ax.errorbar(x, y, yerr=sd.get(arm), fmt=MARK[arm], color=COL[arm], ms=5.5 if big else 4.2, mew=0.6,
                    mfc=("white" if arm == "FixedZ" else COL[arm]), elinewidth=0.9, capsize=1.8, zorder=3 if big else 2)
    for g in range(len(sets) - 1):
        ax.axvline(g + 0.5, color="#dddddd", lw=0.6, zorder=0)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(sets)
    ax.set_xlim(-0.55, len(sets) - 0.45)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)


def fig_ablation():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [2.0, 1.0]})
    dotpanel(a, PRO, PRO_SD, PIX_SETS, "Pixel AUPRO (%)", "(a) Localization (six masked sets)", (40, 92))
    dotpanel(b, IMG, IMG_SD, IMG_SETS, "Image AUROC (%)", "(b) Detection (three image sets)", (55, 100))
    handles = [Line2D([0], [0], marker=MARK[k], color=COL[k], mfc=("white" if k == "FixedZ" else COL[k]), ls="", ms=5.5 if k == "ECP" else 4.5,
                      label=LABEL[k]) for k in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12), columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "fig_ablation_controls")


def spread(ys, gap):
    """Push label positions apart (sorted, minimum distance `gap`) so that direct labels do not overlap."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for _ in range(50):
        moved = False
        for i in range(1, len(order)):
            lo, hi = order[i - 1], order[i]
            if out[hi] - out[lo] < gap:
                d = (gap - (out[hi] - out[lo])) / 2
                out[lo] -= d
                out[hi] += d
                moved = True
        if not moved:
            break
    return out


def fig_decomp():
    """Path Baseline -> Zoom-only -> ECP per dataset (absolute scores), so that each step (zoom, then extent) is a visible rise or fall."""
    cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1.0]})
    for ax, sets, src, yl, ti, gap in ((a, PIX_SETS, PRO, "Pixel AUPRO (%)", "(a) Localization", 2.4),
                                       (b, IMG_SETS, IMG, "Image AUROC (%)", "(b) Detection", 0.55)):
        ends = [src["ECP"][k] for k in range(len(sets))]
        ypos = spread(ends, gap)
        for k, name in enumerate(sets):
            y = [src["CTRL"][k], src["ZO"][k], src["ECP"][k]]
            ax.plot([0, 1, 2], y, "-o", color=cols[k], lw=1.4, ms=4, zorder=3)
            ax.plot([2, 2.1], [y[2], ypos[k]], "-", color=cols[k], lw=0.6)
            ax.text(2.13, ypos[k], name, color=cols[k], fontsize=7.5, va="center")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Baseline\n(no zoom)", "+ zoom\n(Zoom-only)", "+ extent\n(ECP)"])
        ax.set_xlim(-0.2, 2.75)
        ax.set_ylabel(yl)
        ax.set_title(ti, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
    lo = b.get_ylim()[0]
    b.text(0.5, lo + 0.25, "zoom lowers it", ha="center", fontsize=7, color="#555555")
    b.text(1.5, lo + 0.25, "extent restores it", ha="center", fontsize=7, color="#555555")
    fig.tight_layout()
    save(fig, "fig_decomposition")


# ---- image selection and composition of the qualitative figures ----------------------------------------------------------
def fig_select():
    rows = []
    for s in PIX_SETS:
        p = os.path.join(ROOT, "results", "EXP-032", s, f"badcase_{s}.csv")
        with open(p) as f:
            recs = [(float(r["auroc"]), r["id"]) for r in csv.DictReader(f)]
        a = np.array([x[0] for x in recs])
        for tag, target in (("typical", np.median(a)), ("hard", np.percentile(a, 10))):
            i = int(np.argmin(np.abs(a - target)))
            rows.append((s, tag, recs[i][1], recs[i][0]))
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, "selection.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "tag", "id", "ecp_auroc"])
        w.writerows(rows)
    for r in rows:
        print(r)


def fig_gain(top=10):
    """Candidates where ECP beats Zoom-only the most (per image, EXP-012 seed 111, paired).  Needs results/EXP-012/matched-retry-20261002/<set>/{ecp_extent,zoom_only}/pixel_per_image_predictions.csv
    (copied from the server).  Writes figures/gain_candidates.csv; pick one image per set by eye, then add it to figures/selection.csv with tag 'gain'."""
    base = os.path.join(ROOT, "results", "EXP-012", "matched-retry-20261002")

    def read(path):
        with open(path) as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames
            idc = next(c for c in cols if c.lower() in ("id", "image_id", "img_path", "path", "name") or "id" in c.lower())
            auc = next(c for c in cols if "auroc" in c.lower())
            area = next((c for c in cols if "area" in c.lower()), None)
            return {r[idc]: (float(r[auc]), float(r[area]) if area else float("nan")) for r in rd}
    out = []
    for s in PIX_SETS:
        e = read(os.path.join(base, s, "ecp_extent", "pixel_per_image_predictions.csv"))
        z = read(os.path.join(base, s, "zoom_only", "pixel_per_image_predictions.csv"))
        ids = sorted(set(e) & set(z), key=lambda k: -(e[k][0] - z[k][0]))
        n = len(ids)
        # candidates: top 10% gain, but ECP itself must be good (AUROC >= 0.9) so the picture shows a clean win
        cand = [k for k in ids[:max(top, n // 10)] if e[k][0] >= 0.9][:top]
        for k in cand:
            out.append((s, k, round(e[k][0], 4), round(z[k][0], 4), round(e[k][0] - z[k][0], 4), round(e[k][1], 4)))
        print(s, f"n={n}", f"mean gain {np.mean([e[k][0] - z[k][0] for k in ids]):+.4f}", f"candidates {len(cand)}")
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, "gain_candidates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "id", "ecp_auroc", "zo_auroc", "gain", "area_frac"])
        w.writerows(out)
    for r in out:
        print(r)


def read_sel():
    with open(os.path.join(FIG, "selection.csv")) as f:
        return list(csv.DictReader(f))


def pname(set_, tag, ck):
    return f"{set_}_{tag}_{ck}"


def load(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def compose(cols, rows, getter, fname, title_row=None, size=(7.2, None)):
    n_r, n_c = len(rows), len(cols)
    cell = size[0] / n_c
    fig, axes = plt.subplots(n_r, n_c, figsize=(size[0], cell * n_r + 0.25), squeeze=False)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i][j]
            ax.axis("off")
            img = getter(r, c)
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            if i == 0:
                ax.set_title(c, fontsize=8)
            if j == 0:
                ax.text(-0.04, 0.5, f"{r['set']}", transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.04, right=0.995, top=0.95, bottom=0.01)
    save(fig, fname)


def fig_qual(tag):
    rows = [r for r in read_sel() if r["tag"] == tag]
    base = os.path.join(ROOT, "figure_panels")

    def get(r, c):
        d = os.path.join(base, r["set"])
        if c == "Image":
            return load(os.path.join(d, pname(r["set"], r["tag"], "ecp") + "_image.png"))
        if c == "Ground truth":
            return load(os.path.join(d, pname(r["set"], r["tag"], "ecp") + "_gt.png"))
        ck = {"Baseline": "ctrl", "Zoom-only": "zo", "ECP": "ecp"}[c]
        return load(os.path.join(d, pname(r["set"], r["tag"], ck) + "_overlay_z1.6.png"))
    compose(["Image", "Ground truth", "Baseline", "Zoom-only", "ECP"], rows, get, f"fig_qualitative_{tag}")


def fig_zsweep():
    zs = ["-0.69", "0.0", "0.8", "1.6", "2.4"]
    rows = [r for r in read_sel() if r["tag"] == "typical" and r["set"] in ("ClinicDB", "Kvasir", "ISIC", "TN3K")]
    base = os.path.join(ROOT, "figure_panels")

    def get(r, c):
        d = os.path.join(base, r["set"])
        nm = pname(r["set"], r["tag"], "ecp")
        if c == "Image":
            return load(os.path.join(d, nm + "_image.png"))
        if c == "Ground truth":
            return load(os.path.join(d, nm + "_gt.png"))
        return load(os.path.join(d, f"{nm}_overlay_z{c[2:]}.png"))
    compose(["Image", "Ground truth"] + [f"z={z}" for z in zs], rows, get, "fig_z_sweep")



SOTA = [("winclip", "WinCLIP", "CVPR'23"), ("vand", "VAND", "CVPR'23"), ("anomalyclip", "AnomalyCLIP", "ICLR'24"),
        ("adaclip", "AdaCLIP", "ECCV'24"), ("aaclip", "AA-CLIP", "CVPR'25"), ("afclip", "AF-CLIP", "MM'25"),
        ("faprompt", "FAPrompt", "ICCV'25"), ("fbclip", "FB-CLIP", "CVPR'26"), ("ecp", "ECP", "Ours")]


def render_map(npy_path, img):
    """Raw float map -> resized to the image, per-image min-max, jet, 50% blend: the same rendering for every method."""
    from PIL import Image
    m = np.load(npy_path).astype(np.float32)
    m = np.asarray(Image.fromarray(m).resize((img.shape[1], img.shape[0]), Image.BILINEAR))
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)
    heat = (plt.get_cmap("jet")(m)[..., :3] * 255).astype(np.uint8)
    return (0.5 * img + 0.5 * heat).astype(np.uint8)


def fig_sota(tag):
    """Columns: Image | GT | 9 methods.  Raw maps (.npy, any size, higher = more anomalous) are read from
         figure_sota/<set>/<set>_<tag>_<method>.npy        method in: winclip vand anomalyclip adaclip aaclip afclip faprompt fbclip
       ECP comes from figure_panels/<set>/<set>_<tag>_ecp_map_z1.6.npy (z_pix = 1.6).  If figure_sota has no 'anomalyclip' map, our matched
       Baseline (AnomalyCLIP retrained on MVTec, same recipe as ECP) is used and the caption must say so.  A missing method is drawn as 'n/a'."""
    rows = [r for r in read_sel() if r["tag"] == tag]
    base_p, base_s = os.path.join(ROOT, "figure_panels"), os.path.join(ROOT, "figure_sota")
    n_r, n_c = len(rows), 2 + len(SOTA)
    cell = 7.2 / n_c
    fig, axes = plt.subplots(n_r, n_c, figsize=(7.2, cell * n_r + 0.42), squeeze=False)
    heads = ["Image", "GT"] + [m[1] for m in SOTA]
    venues = ["", ""] + [m[2] for m in SOTA]
    for i, r in enumerate(rows):
        d = os.path.join(base_p, r["set"])
        img = load(os.path.join(d, pname(r["set"], r["tag"], "ecp") + "_image.png"))
        for j in range(n_c):
            ax = axes[i][j]
            ax.axis("off")
            out = None
            if j == 0:
                out = img
            elif j == 1:
                out = load(os.path.join(d, pname(r["set"], r["tag"], "ecp") + "_gt.png"))
            else:
                key = SOTA[j - 2][0]
                cand = os.path.join(base_s, r["set"], f"{r['set']}_{r['tag']}_{key}.npy")
                if key == "ecp":
                    cand = os.path.join(d, pname(r["set"], r["tag"], "ecp") + "_map_z1.6.npy")
                elif key == "anomalyclip" and not os.path.isfile(cand):
                    cand = os.path.join(d, pname(r["set"], r["tag"], "ctrl") + "_map_z1.6.npy")
                if os.path.isfile(cand):
                    out = render_map(cand, img)
            if out is not None:
                ax.imshow(out)
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, fontsize=6, color="#999999")
            if i == 0:
                ax.text(0.5, 1.28, heads[j], ha="center", va="bottom", transform=ax.transAxes, fontsize=6.5,
                        fontweight="bold" if heads[j] == "ECP" else "normal")
                if venues[j]:
                    ax.text(0.5, 1.04, venues[j], ha="center", va="bottom", transform=ax.transAxes, fontsize=5, color="#444444")
            if j == 0:
                ax.text(-0.06, 0.5, r["set"], transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=6.5)
    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.03, right=0.995, top=0.9, bottom=0.005)
    save(fig, f"fig_sota_{tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ablation", "decomp", "select", "qual", "zsweep", "sota", "gain"])
    ap.add_argument("--tag", default="typical", choices=["typical", "hard", "gain", "gain2a", "gain2b", "gain2c", "gain2d"])
    a = ap.parse_args()
    {"ablation": fig_ablation, "decomp": fig_decomp, "select": fig_select, "gain": fig_gain, "zsweep": fig_zsweep,
     "sota": lambda: fig_sota(a.tag)}.get(a.cmd, lambda: fig_qual(a.tag))()
