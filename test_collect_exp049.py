"""python test_collect_exp049.py  (numpy/scipy only). Known-answer tests of the EXP-049 rules R1/R2, the pooled-parity helper, and the INCOMPLETE paths on synthetic CSVs."""
import os
import csv
import tempfile
import numpy as np
import pe_shift_stats as S
import analyze_pe_shift as A
import collect_exp049 as K

M, N, X, I, C = "PE-MEDIATED", "NOT PE-MEDIATED", "MIXED", "INCONCLUSIVE", "INCOMPLETE"


def lab(**kw):
    base = {t: N for t in K.ORDER}
    base.update(kw)
    return base


def test_r1():
    assert K.r1_verdict(lab(ecp_s222=M, ecp_s333=M)) == ("REPLICATED", 2)
    assert K.r1_verdict(lab(ecp_s222=M, ecp_s333=X)) == ("MIXED", 1)
    assert K.r1_verdict(lab(ecp_s222=I, ecp_s333=M)) == ("MIXED", 1)
    assert K.r1_verdict(lab(ecp_s222=X, ecp_s333=N)) == ("NOT REPLICATED", 0)
    assert K.r1_verdict(lab(ecp_s222=M, ecp_s333=C))[0] == "INCOMPLETE"
    assert K.r1_verdict({"ecp_s222": M})[0] == "INCOMPLETE"                         # s333 absent
    assert K.r1_verdict(lab(ecp_s222=M, ecp_s333=M, ecp_s111=N)) == ("REPLICATED", 2)   # s111 does not enter the rule


def test_r2():
    assert K.r2_verdict(lab(ctrl_s111=M, ctrl_s222=M, zo_s222=M, zo_s333=M))[0] == "GENERAL"           # exactly 2/3 each
    assert K.r2_verdict(lab(ctrl_s111=M, ctrl_s222=M, ctrl_s333=M, zo_s111=M, zo_s222=M, zo_s333=M))[0] == "GENERAL"
    assert K.r2_verdict(lab(ctrl_s111=M, ctrl_s222=M, zo_s333=M))[0] == "PARTIAL"                     # CTRL 2/3, ZO exactly 1/3
    assert K.r2_verdict(lab(zo_s111=M, zo_s222=M, ctrl_s333=M))[0] == "PARTIAL"
    assert K.r2_verdict(lab(ctrl_s111=M, zo_s333=M))[0] == "NOT GENERAL"                              # 1/3 and 1/3
    assert K.r2_verdict(lab())[0] == "NOT GENERAL"
    # INCONCLUSIVE and MIXED count as not mediated
    lb = lab(ctrl_s111=M, ctrl_s222=I, ctrl_s333=X, zo_s111=M, zo_s222=M)
    r, c = K.r2_verdict(lb)
    assert r == "PARTIAL" and c == {"CTRL": 1, "ZO": 2}
    assert K.r2_verdict(lab(ctrl_s111=M, ctrl_s222=M, zo_s111=M, zo_s222=M, zo_s333=C))[0] == "INCOMPLETE"
    assert K.r2_verdict({t: M for t in K.ORDER if not t.startswith("zo")})[0] == "INCOMPLETE"


def test_pooled_parity():
    assert S.pooled_parity(0.8830, 88.3)[1] and S.pooled_parity(0.8845, 88.3)[1]              # exactly 0.15 passes
    assert not S.pooled_parity(0.8846, 88.3)[1] and not S.pooled_parity(0.8814, 88.3)[1]
    assert not S.pooled_parity(float("nan"), 88.3)[1]
    assert set(A.RECORDED_POOLED_CLINICDB) == set(K.ORDER) and A.RECORDED_POOLED_CLINICDB["ctrl_s333"] == 81.8 and A.RECORDED_POOLED_CLINICDB["ecp_s222"] == 87.6


def write_set(d, s, n, d_img, d_pe, g, seed=0):
    """Synthetic peripheral tercile: sham ~ 0.8; img_rec = sham + d_img; pe_rec = sham + d_pe; pe_rand = pe_rec - g; plus a small central group."""
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.join(d, s), exist_ok=True)
    with open(os.path.join(d, s, f"pe_{s}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(A.COLS)
        for k in range(n):
            sh = 0.8 + 0.02 * rng.randn()
            vals = dict(id=f"{s}{k}", group="peripheral", c=0.8, ci=1, cj=1, sy_rec=1, sx_rec=1, sy_rand=-1, sx_rand=-1, n_les_reg=100, n_bg_reg=100, status="kept",
                        auroc_full_sham=sh, auroc_ref=float("nan"), auroc_sham=sh, auroc_pe_rec=sh + d_pe + 0.005 * rng.randn(), auroc_img_rec=sh + d_img + 0.005 * rng.randn(),
                        auroc_img_rand=sh - 0.05, auroc_pe_flat=sh - 0.1)
            vals["auroc_pe_rand"] = vals["auroc_pe_rec"] - g
            w.writerow([vals.get(c, float("nan")) for c in A.COLS])
        for k in range(10):
            sh = 0.85
            w.writerow([dict(id=f"c{k}", group="central", c=0.1, ci=1, cj=1, sy_rec=0, sx_rec=0, sy_rand=0, sx_rand=0, n_les_reg=100, n_bg_reg=100, status="kept", auroc_full_sham=sh,
                             auroc_ref=float("nan"), auroc_sham=sh, auroc_pe_rec=sh + 0.005).get(c, float("nan")) for c in A.COLS])


def test_checkpoint_labels_and_incomplete():
    with tempfile.TemporaryDirectory() as td:
        root, e48 = os.path.join(td, "e49"), os.path.join(td, "e48")
        for s in S.DECISION:                       # PE-mediated: R = 0.08/0.05 = 1.6, G = 0.1
            write_set(os.path.join(root, "ecp_s222"), s, 60, 0.05, 0.08, 0.10)
        for s in S.DECISION:                       # not mediated: R = 0.005/0.05 = 0.1 (<= 0.20 -> NOT) with G ~ 0
            write_set(os.path.join(root, "ctrl_s111"), s, 60, 0.05, 0.005, 0.0)
        for s in S.DECISION[:2]:                   # third decision set missing
            write_set(os.path.join(root, "zo_s111"), s, 60, 0.05, 0.08, 0.10)
        for s in S.DECISION:                       # D_IMG below 0.02 on all sets -> INCONCLUSIVE
            write_set(os.path.join(root, "zo_s222"), s, 60, 0.005, 0.01, 0.0)
        a = K.load_checkpoint("ecp_s222", root, e48, 200)
        assert a["label"] == M and a["counts"] == (3, 3, 0), a
        b = K.load_checkpoint("ctrl_s111", root, e48, 200)
        assert b["label"] == N, b["label"]
        c = K.load_checkpoint("zo_s111", root, e48, 200)
        assert c["label"] == "INCOMPLETE" and len(c["missing"]) == 1
        d = K.load_checkpoint("zo_s222", root, e48, 200)
        assert d["label"] == I, d["label"]
        assert K.load_checkpoint("zo_s333", root, e48, 200)["label"] == "INCOMPLETE"          # nothing at all
        # ecp_s111 is read from the EXP-048 folder
        for s in S.DECISION:
            write_set(e48, s, 60, 0.05, 0.08, 0.10)
        assert K.load_checkpoint("ecp_s111", root, e48, 200)["label"] == M
        # the same data the EXP-048 collector would use gives the same statistics (verdict function unchanged)
        rows = A.load_rows(os.path.join(root, "ecp_s222", "ClinicDB", "pe_ClinicDB.csv"))
        st, _ = A.set_stats(rows, 200)
        assert S.verdict({s: st for s in S.DECISION})[0] == M


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print("ok", n)
