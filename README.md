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
own. The shipped system uses both: the learned model for its lower mean
error and better accuracy at the trained ~10x scale, plus a local NCC
"snap" step to recover precision the model's coarse response grid can't
reach on its own (NCC's exhaustive per-pixel search has no quantization
floor). Classical NCC still wins outright on a scale the model never
trained on (see the scale-variation results) - this is a partially-solved
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
- `--boundary-bias P` - probability of biasing the reference crop to
  straddle a mat/strip zone boundary (default 0.35) -- the actual
  disambiguating signal, see "Current results" below
- `--mag-ratio R` / `--mag-ratio-jitter-pct P` - base magnification ratio
  (default 10, matching the problem statement) and how far to jitter it
  per-pair, for building a scale-variation robustness test set (see
  `data/scale_variation_eval`)
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

### What the self-eval set is testing, and why

In response to the organizers' own guidance on Track 2 (no fixed dataset
size; at least 30 representative cases; diversity over quantity;
explicitly justify what each case evaluates), `data/self_eval`'s 40 pairs
and the supplementary `data/scale_variation_eval`'s 24 pairs are built
around specific, named failure modes rather than being an arbitrary
random sample:

| Dimension | How it's covered | What it stresses |
|---|---|---|
| Architecture balance | 20 DRAM / 20 FinFET, spanning all 12 presets | The algorithm isn't tuned to one die family |
| Noise | All pairs use the "harder" noise tier (1.8-2.6x the training range) | Robustness to worse-than-training degradation, per the mandatory "test set will be noisier" requirement |
| Rotation | 0.03-2.98 degrees of independent search-side jitter | Independent-capture geometric drift between reference and search |
| Repetitive patterns | Every pair is a periodic DRAM/FinFET array by construction | The core structural difficulty of this problem |
| Genuinely ambiguous cases | 9/40 (22.5%) pairs have zero disambiguating shading/boundary cue | The problem statement's explicit requirement for "at least one highly periodic array region where correct localization is genuinely difficult" -- these are expected failures, and `evaluate.py` reports them separately (`ambiguous_periodic` field) rather than hiding them in the aggregate |
| Scale variation | Separate `data/scale_variation_eval` set: magnification ratio drawn from 8x-12x per pair (`--mag-ratio-jitter-pct 20`), instead of always exactly the problem statement's ~10x | Whether the localization approach generalizes to a *known but non-standard* magnification, not just the one ratio it was implicitly tuned around. Both `inference.py`/`localize()` and `baseline_ncc.py` take the true ratio as an explicit parameter (never inferred), so this tests correctness of that scale-handling logic specifically, isolated from the noise/pattern-matching problem |

Every pair's manifest record carries the ground-truth parameters that
produced it (`ambiguous_periodic`, `mag_ratio`, `family`, full noise/
jitter params), so any accuracy number in this repo can be broken down
by exactly which of these dimensions it came from, rather than reported
only as one aggregate.

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

Two more stages sit after this in `localize()`: a `SubpixelHead` regresses
a bounded correction from the local response-map neighborhood around the
peak (present in the architecture, currently shipped zeroed - see
"Current results"), and a final local-NCC "snap" confined to a small
window around the model's prediction, which *is* enabled and does help
(same section).

## Current results & honest limitations

The checkpoint in `weights/driftsense.pt` is v5's encoder/correlation/
head weights (config `dram_finfet_v5`, 150 epochs on 800 training pairs)
-- still the best-performing of six training attempts on this cluster --
plus a sub-pixel refinement head (see below) that ships mathematically
zeroed (a true, verified no-op) because a later retraining attempt to
actually train it did not improve on v5. Self-evaluated on 40 held-out,
harder-noise pairs (`data/self_eval`):

| Metric | DriftSense (learned + NCC snap) | Classical NCC baseline (`baseline_ncc.py`) |
|---|---|---|
| Accuracy within 5px | 7.5% | 7.5% |
| Accuracy within 30px | 12.5% | 12.5% |
| Accuracy within 100px | 22.5% | 17.5% |
| Mean error | 322 px | 413 px |
| Inference time | ~50 ms/pair (CPU) | ~850 ms/pair (CPU, multi-scale) |

Plus a dedicated **scale-variation** test (`data/scale_variation_eval`,
24 pairs, magnification ratio drawn 8x-12x per pair instead of always
~10x -- see "What the self-eval set is testing" above):

| Metric | DriftSense (learned + NCC snap) | Classical NCC baseline |
|---|---|---|
| Accuracy within 100px | 4.2% | 33.3% |
| Mean error | 349 px | 342 px |

**This is not a solved localization problem**, and it is a genuinely
mixed result, not a clean win for either method: the learned model is
noticeably better at the standard ~10x scenario it was trained on
(22.5% vs. 17.5% @100px, ~25% lower mean error), but classical NCC -- via
its parameter-free multi-scale sweep, with no learned scale-specific bias
-- generalizes far better to magnification ratios the model never trained
on (33.3% vs. 4.2% @100px). Reporting the full picture rather than the
flattering half of it:

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
6. **v6** -- targeted precision fixes suggested by an external review of
   v5's diagnostics: a `SubpixelHead` regressing a bounded correction from
   the local response-map neighborhood around the soft-argmax peak
   (addresses the 16px quantization floor below), a post-hoc local-NCC
   "snap" step confined to a small window around the model's own
   prediction, and training data scaled 800 -> 5000 pairs. Also fixed a
   real infrastructure bug found while investigating a 10x per-epoch
   slowdown on the shared `dgx-a100-02` node: the job requested 8 CPUs
   while the config asked for 12 dataloader workers; fixed the allocation
   and added in-RAM image caching (`training/dataset.py`), cutting epoch
   time from 429s to ~30-40s. Result, measured rigorously rather than
   assumed: retraining with the sub-pixel head (330 total epochs, 5000
   pairs) did **not** beat v5 -- worse at every tolerance (12.5% vs.
   22.5% @100px) despite far more data and compute, most likely because
   batch size was doubled (16->32) without adjusting the learning rate,
   not because the head is unsound. Separately, testing the NCC-snap idea
   in clean isolation (zeroing the subpixel head's output layer -- a
   *provable* no-op, not just an empirically small one -- and re-running
   on v5's unmodified predictions) showed it **does** genuinely help:
   0%->7.5% within 5px, 7.5%->12.5% within 30px, unchanged at 100px, mean
   error statistically unchanged. That clean result is what's shipped.
   An earlier, sloppier test that seemed to show NCC-snap hurting was
   confounded by testing it against an *un-zeroed*, randomly-initialized
   subpixel head instead of clean predictions -- a reminder that an
   ablation needs every other variable actually held fixed.

### Why the gap to the reference scaffold's 60-75% remains

1. **Simplified zone variation**: the reference scaffold draws an
   independent preset for *every* mat cell; this generator uses a
   2-preset checkerboard (`patterns._mat_cell_parity`) to stay
   implementable in the time available, leaving the zone pattern itself
   coarsely periodic (repeats every ~2 mat cells instead of never).
2. **Coarse-to-fine quantization floor**: the coarse response map has a
   16px cell stride (`ENCODER_STRIDE`), so 5px-tolerance accuracy is
   structurally hard for this architecture regardless of training quality.
   The shipped NCC-snap step recovers some of this (see v6 above), but a
   genuinely trained sub-pixel head remains unvalidated -- v6's attempt
   was confounded by an unrelated optimizer/batch-size regression, not
   evidence the idea itself doesn't work.
3. **Unverified exact calibration**: dose/noise magnitudes were tuned by
   direct measurement on our own data (see `docs/CITATIONS.md`'s NCC
   true-vs-random-score diagnostic), not copied from the scaffold's exact
   `GenerationParams` defaults, which weren't fully recovered from the
   fetched source.
4. **No generalization to novel scale**: the model was only ever trained
   at ~10x magnification; the scale-variation test above shows it doesn't
   zero-shot generalize to 8x-12x nearly as well as parameter-free
   classical NCC does, even though `localize()` is given the true ratio
   explicitly. Training data would need to actually include magnification
   jitter (`--mag-ratio-jitter-pct`, already implemented) for this to
   improve -- not attempted here given cluster time budget.

### What would close the remaining gap, in priority order

1. **Re-run the v6 sub-pixel-head training with matched hyperparameters**
   (batch size 16, not 32, or a scaled-up learning rate) to separate the
   head's real effect from the optimizer regression that confounded v6.
2. **Full per-mat-cell random preset draws** instead of a 2-preset
   checkerboard, matching the reference scaffold's zone generator exactly.
3. **Train with magnification-ratio jitter included**
   (`--mag-ratio-jitter-pct`, already implemented in the generator) so the
   model actually sees scale variation during training, not just at
   inference.
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
