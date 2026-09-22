"""
Aggregate pixel_auroc/pixel_aupro across seeds from test.py log files, report mean +/- std.

Looks for log.txt under each --root/<cfg>_s<seed>_sigma<sigma>/<dataset>/ (seed 222, 333) and, for
seed 111, under the legacy paths --root111 (result_cmp for sigma=4, result_cmp32 for sigma=32) that
were produced before this seed-sweep existed. Both layouts share the same "objects | pixel_auroc |
pixel_aupro" table printed at the end of every test.py log.
"""
import os
import re
import argparse
import numpy as np

PAT = re.compile(r"\|\s*colon\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")


def read_metrics(log_path):
    if not os.path.isfile(log_path):
        return None
    text = open(log_path).read()
    matches = PAT.findall(text)
    if not matches:
        return None
    auroc, aupro = matches[-1]  # last table in the file = the final metrics block
    return float(auroc), float(aupro)


def main(args):
    datasets = ["CVC-ClinicDB", "Kvasir", "CVC-ColonDB"]
    configs = ["ctrl", "zoom"]
    sigmas = [4, 32]
    seed111_dir = {4: "result_cmp", 32: "result_cmp32"}

    data = {}  # (cfg, sigma, ds) -> list of (seed, auroc, aupro)
    for cfg in configs:
        for sigma in sigmas:
            for ds in datasets:
                vals = []
                p111 = os.path.join(seed111_dir[sigma], cfg, ds, "log.txt")
                m = read_metrics(p111)
                if m:
                    vals.append((111, *m))
                for seed in (222, 333):
                    p = os.path.join(args.root, f"{cfg}_s{seed}_sigma{sigma}", ds, "log.txt")
                    m = read_metrics(p)
                    if m:
                        vals.append((seed, *m))
                data[(cfg, sigma, ds)] = vals

    for sigma in sigmas:
        print(f"\n##### sigma = {sigma} #####")
        print(f"{'dataset':<14}{'cfg':<6}{'seeds':<14}{'AUROC mean±std':<18}{'PRO mean±std':<18}")
        for ds in datasets:
            for cfg in configs:
                vals = data[(cfg, sigma, ds)]
                if not vals:
                    print(f"{ds:<14}{cfg:<6}{'(missing)':<14}")
                    continue
                seeds = [v[0] for v in vals]
                auroc = np.array([v[1] for v in vals])
                pro = np.array([v[2] for v in vals])
                print(f"{ds:<14}{cfg:<6}{str(seeds):<14}"
                      f"{auroc.mean():>5.2f} +/- {auroc.std(ddof=1) if len(auroc) > 1 else 0:<6.2f}"
                      f"{'':<3}{pro.mean():>5.2f} +/- {pro.std(ddof=1) if len(pro) > 1 else 0:<6.2f}")
            c, z = data[("ctrl", sigma, ds)], data[("zoom", sigma, ds)]
            if len(c) > 1 and len(z) > 1:
                da = np.array([v[1] for v in z]).mean() - np.array([v[1] for v in c]).mean()
                dp = np.array([v[2] for v in z]).mean() - np.array([v[2] for v in c]).mean()
                # pooled std of the difference of two independent means
                sa = np.sqrt(np.var([v[1] for v in c], ddof=1) / len(c) + np.var([v[1] for v in z], ddof=1) / len(z))
                sp = np.sqrt(np.var([v[2] for v in c], ddof=1) / len(c) + np.var([v[2] for v in z], ddof=1) / len(z))
                print(f"{'':<14}{'diff':<6}{'':<14}{da:>+5.2f} (se {sa:.2f}){'':<3}{dp:>+5.2f} (se {sp:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="result_seeds")
    main(ap.parse_args())
