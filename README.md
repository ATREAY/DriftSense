# DriftSense - Navigation-Error Recovery for Semiconductor Wafer Inspection

A template-localization system for the Applied Materials hackathon track
"Drift-Sense: Navigation-Error Recovery." Given a high-magnification
**reference** image of a DRAM-style or FinFET-style die region and a
lower-magnification **search** image covering a ~10x larger physical
area, DriftSense predicts the pixel `(x, y)` center where the reference
pattern reappears (shrunk ~10x) inside the search image.

This is **not** an image-restoration task (see the related `WaferRestore`
project for that KLA problem) it's template matching / localization
under realistic SEM degradation and scale mismatch.

## Why this approach (not simple template matching)

Classical normalized cross-correlation (NCC) template matching struggles
on these images for two structural reasons: (1) DRAM/FinFET layouts are
near-perfectly periodic, so NCC produces many equally-strong peaks at
every lattice period unless the local content it's matching against is
genuinely non-repeating; and (2) the ~10x scale gap between reference and
search means naive same-scale correlation is meaningless without first
resampling to a consistent scale. DriftSense's learned model addresses
(2) directly (the reference is pre-resized by the known magnification
ratio before correlation) and was intended to address (1) via learned,
noise-robust features with a large receptive field.

**In practice, (1) turned out to be primarily a *dataset* problem, not an
architecture one** - see "Current results" below. A single uniform
periodic pattern ties every method, learned or classical, at every
lattice period; what actually breaks the tie is non-periodic structure in
the scene itself (real chips are built from periodic array "mats"
separated by non-periodic routing "strips"). Once the dataset generator
modeled that, the classical NCC baseline improved dramatically on its
own, and currently **beats the learned model at tight tolerances**
(NCC's exhaustive per-pixel search has no quantization floor; the
learned model's coarse response grid does). Both results, and the honest
comparison, are in "Current results" - this is a partially-solved
problem with a clearly diagnosed remaining gap, not a finished win for
either approach.

## Repository layout

```
generator/            synthetic dataset generator (DRAM/FinFET, SEM noise, augmentation)
models/                backbone, cross-correlation, full SiameseLocalizer
training/               PyTorch Dataset, losses, train.py, Slurm scripts
inference.py            standalone Phase-2 inference script (reference+search -> x,y)
evaluate.py              self-evaluation: accuracy@tolerance, timing, success/failure viz
configs/                 training configs (YAML)
docs/CITATIONS.md         references for every noise/augmentation/architecture choice
weights/driftsense.pt      trained checkpoint (downloadable, loaded automatically)
evaluation_report/         self-eval results.json + example visualizations
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

(Developed against Python 3.13 / PyTorch 2.1.0+cu121; a CPU-only install
works too, just slower.)

## 1. Generate a dataset

```bash
python generator/generate_dataset.py \
    --architecture both \
    --num-pairs 40 \
    --output-dir data/self_eval \
    --seed 0 --workers 8
```

Key flags:
- `--architecture {dram,finfet,both}`
- `--num-pairs N`
- `--output-dir DIR` - writes `reference/*.png`, `search/*.png`,
  `manifest.json` (ground-truth center + every generation parameter per
  pair, for full reproducibility)
- `--harder-noise` - widen the search-image noise range (used to build
  a validation split that's noisier than training, matching how the
  organizers' held-out test set is described)
- `--ambiguous-prob P` - fraction of samples with ~0 shading cue, i.e.
  the genuinely-hard fully-periodic case the problem statement requires
  at least one instance of
- `--workers N` - CPU-parallel generation (each pair is independent)

On a shared cluster, don't run large batches on the login node - use
`training/slurm_generate.sh <output_dir> <num_pairs> [architecture]`,
which submits a CPU-only Slurm job.

## 2. Train

```bash
python training/train.py --config configs/dram_finfet.yaml
```

Or, on the Slurm cluster, with automatic GPU-tier selection:

```bash
training/slurm_train.sh configs/dram_finfet.yaml
```

### GPU priority: A100 > V100 > P100

`training/select_gpu.sh` queries live idle-GPU counts (via `scontrol show
node`, not just partition state - a node can be `ALLOCATED` while still
having free GPUs) across this cluster's nodes in priority order:

| Priority | Type | Nodes |
|---|---|---|
| 1 | A100 | `dgx-a100-02` |
| 2 | V100 | `dgx-v100-01` |
| 3 | P100 | `dgx-p100-01`, `cse-node009..012` |

It picks the highest-tier node with at least one GPU idle *right now* and
falls through to the next tier only if the current one has none free.
`training/slurm_train.sh` submits the job pinned to that node. If nothing
is free anywhere, it submits to the top-priority (A100) node so the job
queues there rather than idling on a lower tier.

## 3. Run inference (what Applied Materials runs on the test set)

```bash
python inference.py --reference path/to/reference.png --search path/to/search.png
```

Prints the predicted `(x, y)` center. Runs unmodified on CPU or GPU, and
loads `weights/driftsense.pt` automatically if `--weights` isn't given.

## 4. Self-evaluate

```bash
python evaluate.py --data-dir data/self_eval --weights weights/driftsense.pt --tolerance-px 30
```

Reports accuracy within tolerance, mean/median pixel error, and mean/
median per-pair inference time; writes one SUCCESS and one HONEST-FAILURE
visualization (reference, search, predicted vs. true location) to
`evaluation_report/`.

## Architecture

```
reference (1000x1000)              search (1000x1000)
       |                                   |
  resize to 128x128               resize to 512x512 (coarse pass)
  [known 10x mag ratio]           or crop+resize (fine/refine pass)
       |                                   |
       +---------- shared CNN encoder -----+   (stride 16, ~188K params)
       |                                   |
  exemplar feat (8x8)              search feat (32x32)
       \                                  /
        depthwise cross-correlation (SiamFC-style)
                       |
              response map (25x25)
                       |
             3-layer conv refine head
                       |
           heatmap logits -> soft-argmax
                       |
              (x, y) in cell space -> pixels
```

Coarse-to-fine: the coarse pass locates the region on the full search
image; a second pass crops a native-resolution window around that peak
and reruns the same shared encoder for a finer estimate - no extra
parameters, just reuse of the same weights at a different input scale
(both regimes are represented during training, see
`training/dataset.py`).

Training loss combines a Gaussian-target heatmap BCE (dense early-training
signal) with a direct L1 on the soft-argmax coordinate (final precision) -
see `training/losses.py` and `docs/CITATIONS.md`.

## Current results & honest limitations

The checkpoint in `weights/driftsense.pt` (config `dram_finfet_v5`, 150
epochs on 800 training pairs) is the best of five training attempts on
this cluster, self-evaluated on 40 held-out, harder-noise pairs
(`data/self_eval`):

| Metric | DriftSense (learned) | Classical NCC baseline (`baseline_ncc.py`) |
|---|---|---|
| Accuracy within 5px | 0% | 7.5% |
| Accuracy within 100px | 22.5% | 17.5% |
| Mean error | 322 px | 413 px |
| Inference time | ~46 ms/pair (CPU) | ~590 ms/pair (CPU, multi-scale) |

**This is not a solved localization problem**, and unlike the v1-v4
numbers below, the learned model no longer cleanly beats the classical
baseline at every tolerance -- NCC is actually more precise at tight (5px)
tolerance, while the learned model has a lower mean error and edges ahead
at loose (100px) tolerance. Reporting the full, mixed picture rather than
the flattering half of it:

### What we tried, in order

1. **v1** (40 epochs, 400 pairs, fully-randomized per-sample geometry):
   model collapsed to always predicting near the image center regardless
   of input. Diagnosed via an overfit test on 8 samples for 150 epochs
   (train error 6.3 -> 0.3 cells) -- proving the pipeline *can* learn, so
   this was an undertrained local optimum, not a code bug.
2. **v2** (250 epochs, same data): escaped the collapse but overfit
   severely -- train error dropped, validation error didn't move.
3. **v3**: found the smooth shading field used as the positional cue had
   too wide a wavelength relative to the reference's footprint; narrowed
   it. Modest improvement (10% @ 100px vs. ~0%).
4. **v4**: 5x more data (2000 pairs) + 280 total epochs -- no further
   improvement, ruling out "just needs more epochs/pairs" as the
   remaining bottleneck.
5. **v5** -- a structural rebuild after comparing against the official
   reference scaffold (huggingface.co/spaces/aayushraina21/drift-sense-
   synthetic-data), which we had not had access to for v1-v4. Its
   classical ZNCC baseline reaches 60-75% accuracy at <=5px with *no
   training at all*, which prompted a direct diagnostic: on a noise-free
   control image, true-match NCC score (0.987) was statistically
   indistinguishable from the best score at a random offset (0.987) --
   i.e. a single uniform periodic pattern is tied at every lattice
   period, and no amount of noise recalibration or training fixes a tie.
   The actual missing piece, confirmed by reading the scaffold's source:
   real chips are composed of periodic "mat" blocks separated by
   non-periodic "strip" material (routing/peripheral circuitry), and
   reference crops are deliberately biased to straddle that boundary --
   *that* non-periodic structure is what breaks the tie, not noise level
   or a smooth shading cue. Rebuilt around this (`generator/presets.py`
   fixed die families + `patterns.py` mat/strip zone compositing +
   boundary-biased crop placement), which measurably fixed the
   *underlying task*: the classical NCC baseline alone went from ~0% to
   7.5-40% accuracy across tolerance levels on control batches (see
   `docs/CITATIONS.md` for the full mat/strip diagnostic writeup) --
   before any model retraining. The retrained model (v5, above) improved
   over v1-v4 in absolute terms but is still well below the reference
   scaffold's 60-75%.

### Why the gap to the reference scaffold's 60-75% remains

1. **Simplified zone variation**: the reference scaffold draws an
   independent preset for *every* mat cell; this generator uses a
   2-preset checkerboard (`patterns._mat_cell_parity`) to stay
   implementable in the time available, leaving the zone pattern itself
   coarsely periodic (repeats every ~2 mat cells instead of never).
2. **Coarse-to-fine quantization floor**: the coarse response map has a
   16px cell stride (`ENCODER_STRIDE`), so 5px-tolerance accuracy is
   structurally hard for this architecture regardless of training quality
   -- NCC searches every pixel directly and has no such floor, which is
   likely why it now wins at tight tolerance despite having no learned
   features at all.
3. **Unverified exact calibration**: dose/noise magnitudes were tuned by
   direct measurement on our own data (see `docs/CITATIONS.md`'s NCC
   true-vs-random-score diagnostic), not copied from the scaffold's exact
   `GenerationParams` defaults, which weren't fully recovered from the
   fetched source.

### What would close the remaining gap, in priority order

1. **Full per-mat-cell random preset draws** instead of a 2-preset
   checkerboard, matching the reference scaffold's zone generator exactly.
2. **A localization head with finer output stride** (e.g. dilate less
   aggressively, or add a true sub-pixel regression head on top of the
   coarse cell) to remove the 16px quantization floor for tight-tolerance
   scoring.
3. **A classical-NCC hybrid**: fuse the multi-scale NCC score
   (`baseline_ncc.py`, cheap, no training, currently *more* precise than
   the learned model at 5px) as an auxiliary prior into the learned
   heatmap, rather than treating them as competing rather than
   complementary signals.
4. **Orders of magnitude more training pairs** (10K-100K+) -- cheap in
   principle (generation is ~1-2s/pair with 16 workers) but not attempted
   at that scale here given cluster time budget.

The full pipeline (dataset generator, training, coarse-to-fine inference,
self-evaluation, classical-NCC baseline, GPU-priority Slurm submission) is
complete and correct end-to-end -- verified via an overfit test,
ground-truth alignment checks, and direct comparison against the official
reference implementation's source code -- the remaining gap is model
precision at tight tolerances, with a concrete, diagnosed path to closing
it above.

## Citations

See `docs/CITATIONS.md` for the 2-3+ references backing every noise
model, structural parameter, augmentation choice, and architectural
decision, as required by the submission guidelines.
