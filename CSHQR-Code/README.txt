CS-HQR: Chroma-Subsampled Hybrid Quantum Representation
========================================================

Version: 1.0
Date:    2026
Author:  vrushali Nikam <vrushali.nikam@ges-coengg.org>
         Department of Computer Enineering,
         MET BKC Institute of Engineering

Paper:   V. Nikam, "CS-HQR: Chroma-Subsampled Hybrid Quantum Representation
         for Efficient Color Image Encoding",
         Image Processing On Line, 2026.
         https://doi.org/10.5201/ipol

Source:  https://github.com/Vrushali-Nikam/Vrushali-Quantum-Image-Representation

Overview
--------
This software implements the CS-HQR algorithm for quantum color image encoding.
CS-HQR exploits the spatial redundancy of chrominance in natural images by
applying YCbCr color-space conversion followed by 4:2:0 chroma subsampling,
then encoding the luminance and chrominance channels into two separate quantum
circuits:

  Circuit A — Luminance (Y) at full NxN resolution  [16 qubits for 16x16]
  Circuit B — Chrominance (Cb + Cr) at N/2 x N/2   [15 qubits for 8x8]

Results on 6,097 NIST-MNI brain MRI slices (16x16):
  - 54% gate count reduction vs MCQI
  - 72% circuit depth reduction vs MCQI
  - SSIM > 0.9999 (near-lossless reconstruction)

The module also includes 11 baseline quantum image encoding methods for
direct comparison: FRQI, EFRQI, QPIE, NEQR, GQIR, INEQR, TNR, MCQI,
QRMW, DCT-QIR, QLR.

Files
-----
  cshqr.py          Main module: CS-HQR encoding/decoding, all baselines,
                    evaluation metrics
                    - rgb_to_ycbcr()             -> Section 4.1
                    - downsample_420()           -> Section 4.2
                    - encode_cshqr()             -> Algorithm 1
                    - decode_cshqr()             -> Algorithm 2
                    - compute_evaluation_metrics() -> Section 5
                    - encode_frqi(), encode_neqr(), etc. -> Section 2
  run_demo.py       CLI entry point for IPOL online demo
  requirements.txt  Python dependencies
  LICENSE.txt       GNU Affero General Public License v3

Requirements
------------
  Python >= 3.10
  See requirements.txt for package dependencies.

  Install with:
    pip install -r requirements.txt

Usage
-----
  Basic CS-HQR encoding (4:2:0 chroma subsampling, 16x16 default):
    python run_demo.py test_data/brain_mri.png

  Custom subsampling and output path:
    python run_demo.py input.png --subsampling 422 --output recon.png

  Save YCbCr channel images:
    python run_demo.py input.png --channels --output-dir channels/

  Compare CS-HQR against all 11 baseline methods:
    python run_demo.py input.png --compare --output-dir results/

  Full options:
    python run_demo.py --help

Output
------
  <name>_cshqr_recon.png  - Reconstructed RGB image after encoding
  <name>_metrics.json     - 10-parameter evaluation metrics (JSON)
  <name>_Y.png            - Luminance channel (with --channels)
  <name>_Cb.png           - Cb chrominance channel (with --channels)
  <name>_Cr.png           - Cr chrominance channel (with --channels)

License
-------
Copyright (C) 2026 V Nikam<vrushali.nikam@ges-coengg.org>.
Licensed under the GNU Affero General Public License v3.
See LICENSE.txt for the full license text.
