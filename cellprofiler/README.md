# CellProfiler pipelines — CP3D

Two-stage CellProfiler workflow used to convert raw confocal z-stacks of HepG2
spheroids (Akura 384 microplates, Operetta CLS) into per-spheroid morphological
feature tables.

## Pipelines (execute in order)

1. **`1. CP3D - projections.cpproj`** — reads the four-channel confocal z-stacks
   (DNA, ER, AGP, MITO) and generates the maximum-intensity projection (MIP)
   per well and per channel. Outputs 16-bit TIFF MIPs consumed by the next stage.

2. **`2. CP3D - features.cpproj`** — reads the MIPs, segments each spheroid as a
   single object using a nucleus-seeded strategy (Robust Background thresholding
   on the DNA channel, erosion + hole-fill, manual threshold 0.8 on the
   reconstructed cellular envelope, area filter ≥ 50 000 px², dilation), and
   extracts 732 morphological features per spheroid across the four channels
   using MeasureObjectSizeShape, MeasureObjectIntensity, MeasureGranularity,
   MeasureTexture and MeasureColocalization. Feature values are exported as
   CSV by ExportToSpreadsheet and feed into the pycytominer normalisation
   step described in the manuscript Methods and root README.

## Software version

Both pipelines were built and tested with **CellProfiler 4.2.8** on Windows 11.

## Citation

If you use these pipelines, please cite:
Iañez I, *et al.* *3D Cell Painting in HepG2 spheroids for phenotypic drug
discovery.* Commun Biol (2026).