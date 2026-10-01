# AnomalyCLIP — project notes for Claude

Research fork of AnomalyCLIP (ICLR'24 zero-shot anomaly detection), aimed at a Q1/rank-A paper on
**medical** zero-shot anomaly detection (not industrial — see Scope below). User's own repo:
`ngodongnguyen/AnomalyClipV2`.

## Environment

- Development happens locally in this repo (Mac). All training/testing happens on a **remote GPU server**
  the user SSHes into themselves (`ai3@ai3:~/NguyenND/AnomalyClipV2`) — Claude cannot run commands there.
  Sync via `git push`/`git pull`; the user pastes logs back for Claude to interpret.
- Server data root: `/home/ai3/NguyenND/AnomalyClipV2/data/` (see `run_ecp.sh` for exact per-dataset paths).
- Local machine lacks `torch`/`cv2` for the main pipeline; `/Users/nguyen.ngo.1/miniconda3/bin/python` has
  `numpy`/`scipy`/`cv2` and is used to unit-test pure-Python logic (stats helpers, shape descriptors, etc.)
  before shipping code to the server. Always unit-test non-trivial diagnostic logic this way first.

## The module: ECP (Extent-Conditioned Prompts)

`extent_prompt.py` (`ExtentConditioner`, `area_to_z`, `visual_descriptor`, `conditioned_text_features`) shifts
AnomalyCLIP's learnable text-prompt context tokens along a continuous **learned lesion-extent axis** `z`
(trained on MVTec + `--zoom_aug_p` crop augmentation, so the axis spans native-small to zoomed-large).

**Current best, final config — two fixed operating points, one per scoring head, same checkpoint:**
- Pixel/localization head: `z_pix = 1.6` (≈ largest extent seen in training)
- Image/classification head: `z_img = -0.69` (≈ smallest extent seen in training, the `AREA_MIN` clamp)
- `test.py` flags: `--ec_z_img -0.69 --ec_z_pix 1.6` (or `--ec_const_z` for one z on both heads,
  `--ec_oracle` = diagnostic-only GT-extent upper bound). Per-image *estimation* of z was tried and **failed**
  (see Falsified ideas) — both z values are fixed constants, not predicted per image.
- `train.py --extent_cond {extent,global,none}` trains the conditioner; `none` = original AnomalyCLIP.

**Key checkpoints (server, `./checkpoints/<name>/epoch_15.pth`):**
| name | what it is |
|---|---|
| `ctrl_mvtec` | original AnomalyCLIP baseline (`zoom_aug_p=0`, no module) |
| `zoom_p1` | augmentation-only control (`zoom_aug_p=1.0`, no module) — isolates what augmentation alone buys |
| `ecp_extent` (+ `_s222`, `_s333`) | the module, 3 seeds, trained with `zoom_aug_p=0.5` |
| `ecp_global` | Crane-style image-context-conditioned control (no extent supervision) |

`run_ecp.sh` is the single entry point for almost everything (train/test/diagnostics) — read it before writing
a new one-off command; most needs already have a subcommand (`fixed`, `test_cls`, `test_heldout`, `test_visa`,
`diag_cb`, `diag_proto`, `diag_shape`, `calibrate`, ...).

## Scope: medical, not general-purpose

`z_pix=1.6` **hurts** industrial VisA (pixel AUROC −2.5, PRO −7.9 vs ctrl; worst on small-defect classes) because
industrial defects are tiny relative to medical lesions. The paper is framed as **medical ZSAD**; VisA/industrial
results go in limitations, not headline results. Don't re-litigate this — it's a measured, pre-registered result.

## Working discipline (the user enforces this — follow it without being asked)

1. **Pre-register the decision rule before running the experiment.** State in the prior turn what result would
   confirm vs falsify the hypothesis. Never adjust the threshold after seeing the numbers to make it pass.
2. **Report negative/failed results as plainly as positive ones.** Do not spin, hedge, or bury a falsification.
3. **Cheap, no-training diagnostics before any training run.** Most scripts here are `analyze_*.py` test-time
   diagnostics (no gradient steps) — exhaust those before proposing a new trained component.
4. **Unit-test new statistics/metrics on synthetic data with a known answer** before running on real data
   (pattern used throughout: synthetic blobs/masks with known area, shape, or offset).
5. A metric-gaming trick (e.g. a loss that directly optimizes the eval metric's known weakness) is **not** an
   acceptable "module" — the user has explicitly rejected this framing more than once.

## Falsified ideas — do not re-propose without new evidence

Scale-consistency loss · background-contrast scoring · coverage-adaptive sigma · learned multi-scale fusion ·
registers/artifact tokens · over-smoothing/depth homogenization · feature-affinity label propagation ·
prompt-bank / mixture-of-experts fusion (same bottleneck as per-image z estimation, just reparameterized) ·
per-image extent estimation, 2 attempts (backbone-feature regression, anomaly-map proxy) · automatic domain-level
z calibration, 3 attempts (Otsu pixel-map fraction, classification-score-spread ratio) · false-positive-hotspot
attribution to low-level image properties, pre-registered and falsified (specular glare, wet/glossy sheen,
device-burned-in corner text, dark regions, FOV border, strong edges — none explain the hotspots; they are a
semantic confusion, not a pixel-level nuisance).

## Where results live

- `RESULTS_SUMMARY.md` — the master results log (markdown tables, all seeds, all datasets, all falsified ideas
  with numbers). Update this whenever a new result changes the story.
- `results.html` — same content, styled for local viewing (`open results.html`). Keep both in sync.
- Memory (`~/.claude/projects/.../memory/project_ecp_module.md`) has a running technical summary across
  sessions — read it at the start of a new session before assuming context from scratch.
