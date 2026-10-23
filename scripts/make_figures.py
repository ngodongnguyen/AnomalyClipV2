"""Paper figures (no architecture figure).  Local, needs matplotlib / numpy / PIL (system python3 has them).

  python3 scripts/make_figures.py ablation           # Fig. controls: ECP vs ZO / DUAL / SHUF / LATENT / CTRL / fixed-z (numbers from the registry)
  python3 scripts/make_figures.py decomp             # Fig. decomposition: zoom vs extent contribution (EXP-030)
  python3 scripts/make_figures.py select             # pick the images for the qualitative figures (rule below) -> figures/selection.csv
  python3 scripts/make_figures.py qual               # compose Fig. qualitative from the panels made on the server (scripts/figure_batch.py)
  python3 scripts/make_figures.py zsweep             # compose Fig. z-sweep from the same panels

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
LABEL = {"CTRL": "CTRL (no zoom)", "ZO": "ZO (zoom only)", "DUAL": "DUAL (learned const.)", "FixedZ": "Fixed-z training$^{\\dagger}$",
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


def fig_decomp():
    zoom_p = [3.4, 5.2, 2.0, 6.4, 6.9, -0.9]     # ZO - CTRL, PRO
    ext_p = [5.1, 2.4, -0.4, 4.2, 9.6, 3.5]      # ECP - ZO, PRO
    tot_p = [8.5, 7.6, 1.6, 10.6, 16.4, 2.7]     # ECP - CTRL, PRO
    zoom_i = [-3.8, -5.0, -5.5]
    ext_i = [5.4, 7.8, 7.7]
    tot_i = [1.6, 2.8, 2.2]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [2.0, 1.0]})
    for ax, sets, z, e, t, yl, ti in ((a, PIX_SETS, zoom_p, ext_p, tot_p, "$\\Delta$ pixel AUPRO (points)", "(a) Localization"),
                                      (b, IMG_SETS, zoom_i, ext_i, tot_i, "$\\Delta$ image AUROC (points)", "(b) Detection")):
        x = np.arange(len(sets))
        w = 0.34
        ax.bar(x - w / 2, z, w, color=COL["ZO"], label="zoom augmentation (ZO $-$ CTRL)", zorder=2)
        ax.bar(x + w / 2, e, w, color=COL["ECP"], label="extent conditioning (ECP $-$ ZO)", zorder=2)
        ax.plot(x, t, "D", color="black", ms=4.2, label="total (ECP $-$ CTRL)", zorder=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sets)
        ax.set_ylabel(yl)
        ax.set_title(ti, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
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
        ck = {"CTRL": "ctrl", "ZO": "zo", "ECP": "ecp"}[c]
        return load(os.path.join(d, pname(r["set"], r["tag"], ck) + "_overlay_z1.6.png"))
    compose(["Image", "Ground truth", "CTRL", "ZO", "ECP"], rows, get, f"fig_qualitative_{tag}")


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ablation", "decomp", "select", "qual", "zsweep"])
    ap.add_argument("--tag", default="typical", choices=["typical", "hard"])
    a = ap.parse_args()
    {"ablation": fig_ablation, "decomp": fig_decomp, "select": fig_select, "zsweep": fig_zsweep}.get(a.cmd, lambda: fig_qual(a.tag))()
