# Citations

Every noise model, structural parameter, and augmentation choice in
`generator/` and every architectural choice in `models/` is justified
below.

## Dataset generator

### Device structure (`generator/patterns.py`)

1. **S. M. Sze & K. K. Ng**, *Physics of Semiconductor Devices*, 3rd ed.,
   Wiley, 2007. — DRAM 1T1C cell array structure: orthogonal word-line /
   bit-line grid with a storage-node contact at every crossing. Basis for
   `render_dram`'s line-grid + via-dot layout.
2. **C. Auth et al.**, "A 22nm high performance and low-power CMOS
   technology featuring fully-depleted tri-gate transistors," *IEDM*,
   2012. — FinFET layout: dense parallel fins crossed by orthogonal gate
   lines. Basis for `render_finfet`'s fin/gate structure.

### SEM noise model (`generator/noise.py`)

3. **A. Foi, M. Trimeche, V. Katkovnik, K. Egiazarian**, "Practical
   Poissonian-Gaussian Noise Modeling and Fitting for Single-Image
   Raw-Data," *IEEE Trans. Image Processing*, 17(10), 2008. — Signal-
   dependent shot noise (Poisson) + additive read noise (Gaussian) model
   used in `poisson_gaussian_noise`.
4. **J. W. Goodman**, *Speckle Phenomena in Optics: Theory and
   Applications*, SPIE Press, 2007. — Multiplicative speckle noise model
   used in `speckle_noise`.
5. **L. Reimer**, *Scanning Electron Microscopy: Physics of Image
   Formation and Microanalysis*, 2nd ed., Springer, 1998. — SEM edge
   effect: secondary-electron yield (and hence measured brightness) rises
   sharply at topographic edges/feature boundaries. Basis for
   `edge_brighten`'s gradient-magnitude-weighted intensity boost.
6. **M. T. Postek & A. E. Vladar**, "Does your SEM really tell the truth?
   How would you know?," *Proc. SPIE* 8036, 2011. — Noise and artifact
   characteristics specific to semiconductor-metrology SEM imaging;
   motivates using a noisier degradation range for the search image than
   the reference (mandatory dataset requirement) and the low-frequency
   dose/shading field added in `patterns._sample_field`.

### Augmentation (`generator/augment.py`)

7. **C. Shorten & T. M. Khoshgoftaar**, "A survey on Image Data
   Augmentation for Deep Learning," *Journal of Big Data*, 6(60), 2019. —
   Rotation and scale jitter as standard augmentations for acquisition/
   viewpoint-drift robustness; justifies the independent geometric jitter
   applied separately to the reference and search captures.

## Model architecture (`models/`)

8. **L. Bertinetto, J. Valmadre, J. F. Henriques, A. Vedaldi, P. H. S.
   Torr**, "Fully-Convolutional Siamese Networks for Object Tracking,"
   *ECCV Workshops*, 2016 (SiamFC). — Shared-encoder + depthwise cross-
   correlation + response-map design that `models/siamese_localizer.py`
   follows, adapted here with a known fixed magnification ratio (so no
   scale pyramid is needed) and a coarse-to-fine refinement pass.

## Notes on scope

Point-level device defects (missing vias/fins, particle contamination)
were intentionally left out of v1 — the difficulty knob used instead is
the amplitude of the smooth non-periodic shading field (`ambiguous_prob`
in `generate_dataset.py`), which controls how close a sample is to the
fully periodic, position-ambiguous case the problem statement calls out
as a required failure mode to test.
