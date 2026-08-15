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

Classical normalized cross-correlation (NCC) template matching fails on
these images for two reasons this dataset is deliberately built to
expose: (1) DRAM/FinFET layouts are near-perfectly periodic, so NCC
produces many equally-strong peaks at every lattice period the position
is only disambiguated by non-periodic global context (die-level dose/
shading variation), which a local template-matching score can't see but
a CNN with a large receptive field over the whole search image can; and
(2) the ~10x scale gap between reference and search means naive
same-scale correlation is meaningless without first resampling to a
consistent scale. DriftSense addresses both: a shared CNN encoder
(SiamFC-style cross-correlation, see `docs/CITATIONS.md`) gives it
learned, noise-robust local features instead of raw pixel correlation,
and the reference is pre-resized by the known magnification ratio before
correlation so both branches operate at a consistent apparent scale.

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
- `--output-dir DIR` — writes `reference/*.png`, `search/*.png`,
  `manifest.json` (ground-truth center + every generation parameter per
  pair, for full reproducibility)
- `--harder-noise` - widen the search-image noise range (used to build
  a validation split that's noisier than training, matching how the
  organizers' held-out test set is described)
- `--ambiguous-prob P` - fraction of samples with ~0 shading cue, i.e.
  the genuinely-hard fully-periodic case the problem statement requires
  at least one instance of
- `--workers N` - CPU-parallel generation (each pair is independent)

On a shared cluster, don't run large batches on the login node — use
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
node`, not just partition state — a node can be `ALLOCATED` while still
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
and reruns the same shared encoder for a finer estimate — no extra
parameters, just reuse of the same weights at a different input scale
(both regimes are represented during training, see
`training/dataset.py`).

Training loss combines a Gaussian-target heatmap BCE (dense early-training
signal) with a direct L1 on the soft-argmax coordinate (final precision) —
see `training/losses.py` and `docs/CITATIONS.md`.

## Current results & honest limitations

The checkpoint in `weights/driftsense.pt` (config `dram_finfet_v3`, 250
epochs on 400 training pairs) is the best of four training attempts on
this cluster, self-evaluated on 40 held-out, harder-noise pairs
(`data/self_eval`, `--tolerance-px 100`):

| Metric | DriftSense (learned) | Classical NCC baseline (`baseline_ncc.py`) |
|---|---|---|
| Accuracy within 100px | 10% | 2.5% |
| Mean error | 333 px | 599 px |
| Median error | 323 px | 591 px |
| Inference time | ~50 ms/pair (CPU) | ~130 ms/pair (CPU) |

The classical baseline (`skimage.feature.match_template`, reference
downsampled by the known 10x magnification ratio, single best NCC peak)
is included precisely to test the README's opening claim rather than
just assert it: DriftSense's learned model is ~45% lower mean error and
4x the within-tolerance accuracy of plain template matching, consistent
with periodic aliasing genuinely defeating single-peak NCC on this data.
Neither number is competition-ready, but the comparison is real and
reproducible (`python baseline_ncc.py --data-dir data/self_eval`).

**This is not a solved localization problem.** It is meaningfully better
than the untrained/collapsed baseline (see below) but still close to
chance on a 1000x1000 canvas. Reporting this honestly rather than hiding
it:

### What we tried, in order

1. **v1** (40 epochs, 400 pairs): model collapsed to always predicting
   near the image center regardless of input (0% within 30px, error is
   about what a fixed-centroid guess gets against a uniformly-random
   target). Diagnosed via an overfit test on 8 samples for 150 epochs,
   which dropped training error from ~6.3 to ~0.3 cells -- proving the
   cross-correlation/soft-argmax pipeline *can* learn, so this was an
   undertrained local optimum, not a code bug.
2. **v2** (250 epochs, same data, higher heatmap loss weight): escaped
   the collapse (train error dropped substantially) but validation error
   didn't move -- severe overfitting to the 400 training instances.
3. **v3** (dataset fix): found that the shading field used as the primary
   non-periodic positional cue had a wavelength (2500-9000 physical
   units) much larger than the reference's own 1000-unit footprint, so
   the reference couldn't encode a locally matchable "fingerprint" of its
   position. Shortened the wavelength range to 700-1900. Modest
   improvement (10% @ 100px vs. ~0%).
4. **v4** (5x more data -- 2000 pairs -- plus 280 total epochs): did not
   improve on v3; validation error stayed flat to slightly worse,
   indicating the bottleneck is no longer epoch count or raw pair count.

### Root cause

This is a from-scratch Siamese matching network (SiamFC-style
architectures are normally trained on tens of thousands of diverse video
sequences). Every generated pair also has fully randomized DRAM/FinFET
structural parameters, so the network must learn a *generic* "compare any
two patches" strategy, not memorize fixed templates -- a much harder
learning problem than the pair count here can support.

### What would actually fix this, in priority order

1. **Orders of magnitude more training pairs** (10K-100K+), generated on
   a dedicated CPU allocation over hours rather than minutes -- cheap in
   principle since generation is ~0.6s/pair with 16 workers, just not
   attempted at that scale here.
2. **A pretrained visual backbone** (e.g. ImageNet-pretrained ResNet
   features, fine-tuned) instead of a from-scratch 128K-parameter
   encoder, so the network isn't learning low-level texture statistics
   from zero on a few thousand synthetic examples.
3. **Less per-sample structural randomization** (e.g. reuse a smaller
   pool of "die families" across many samples) so the network sees each
   underlying pattern enough times to learn its shading fingerprint,
   closer to how the real deployment scenario would actually work
   (inspecting variations of a small number of known die designs, not an
   unbounded space of random layouts).
4. **A classical-NCC hybrid**: fuse a normalized cross-correlation score
   (cheap, exact, no training needed) as an auxiliary input/prior
   alongside the learned heatmap, rather than relying purely on learned
   features to rediscover what correlation already gives for free.

The full pipeline (dataset generator, training, coarse-to-fine inference,
self-evaluation, GPU-priority Slurm submission) is complete and correct
end-to-end -- verified via an overfit test, ground-truth alignment checks,
and the diagnostic sequence above -- the remaining gap is model accuracy,
addressable with more training-data scale than this session's cluster
time budget covered.

## Citations

See `docs/CITATIONS.md` for the 2-3+ references backing every noise
model, structural parameter, augmentation choice, and architectural
decision, as required by the submission guidelines.
