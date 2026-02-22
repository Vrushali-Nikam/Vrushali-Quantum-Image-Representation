# -*- coding: utf-8 -*-
#
# CS-HQR: Chroma-Subsampled Hybrid Quantum Representation
#         Command-line demo / evaluation driver
#
# Copyright (C) 2026 Raiyyan Patel <raiyyanpatel467@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
"""
run_demo.py — CLI entry point for CS-HQR encoding demonstration.

Usage examples
--------------
Encode a single image and print metrics:

    python run_demo.py brain.png

Encode with explicit 4:2:2 chroma subsampling and save reconstruction:

    python run_demo.py brain.png --subsampling 422 --output recon.png

Compare CS-HQR against all 11 baseline methods:

    python run_demo.py brain.png --compare --output-dir results/

Show YCbCr channel decomposition:

    python run_demo.py brain.png --channels

Options
-------
  --subsampling {420,422,444}
                        Chroma subsampling mode [default: 420]
  --bit-depth INT       Number of color qubits per channel [default: 8]
  --max-size INT        Resize image to at most MAX_SIZE x MAX_SIZE before
                        processing [default: 16]
  --output FILE         Save reconstructed RGB image to FILE
  --output-dir DIR      Save all output files to DIR [default: .]
  --channels            Save individual Y/Cb/Cr channel images
  --compare             Run all 11 baselines and print comparison table
  --json FILE           Write metrics to a JSON file
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with:  pip install Pillow")
    sys.exit(1)

try:
    from cshqr import (
        encode_cshqr,
        decode_cshqr,
        encode_image_all_methods,
        compute_evaluation_metrics,
        rgb_to_ycbcr,
        METHOD_NAMES,
        EVALUATION_PARAMS,
    )
except ImportError as e:
    print(f"ERROR: Cannot import cshqr module — {e}")
    print("Make sure cshqr.py is in the same directory as run_demo.py.")
    sys.exit(1)


# ============================================================================
# IMAGE I/O HELPERS
# ============================================================================

def load_and_preprocess(path, max_size=16):
    """
    Load an image from disk, convert to RGB uint8, and resize to at most
    max_size x max_size (power-of-2 dimensions for quantum position registers).

    Parameters
    ----------
    path : str
        Path to the input image file.
    max_size : int
        Maximum width / height (rounded down to nearest power of 2).

    Returns
    -------
    img : numpy.ndarray
        RGB image of shape (H, W, 3), dtype uint8.
    """
    img = Image.open(path).convert("RGB")

    # Round max_size down to nearest power of 2
    target = 1
    while target * 2 <= max_size:
        target *= 2

    img = img.resize((target, target), Image.LANCZOS)

    print(f"  Input image : {path}")
    print(f"  Loaded size : {img.width} x {img.height} (RGB)")
    return np.array(img, dtype=np.uint8)


def save_image(array_rgb, path):
    """Save an H x W x 3 uint8 numpy array as an image file."""
    Image.fromarray(array_rgb.astype(np.uint8)).save(path)
    print(f"  Saved image : {path}")


def save_channel_images(ycbcr_full, output_dir, prefix):
    """Save Y, Cb, Cr channels as separate grayscale PNG images."""
    os.makedirs(output_dir, exist_ok=True)
    for i, name in enumerate(['Y', 'Cb', 'Cr']):
        ch = np.clip(ycbcr_full[..., i], 0, 255).astype(np.uint8)
        p  = os.path.join(output_dir, f"{prefix}_{name}.png")
        Image.fromarray(ch, mode='L').save(p)
        print(f"  Saved channel {name} : {p}")


# ============================================================================
# DISPLAY / REPORTING
# ============================================================================

def print_metrics(metrics):
    """Pretty-print the 10-parameter metrics dict."""
    method = metrics.get('method', '?')
    print(f"\n{'='*60}")
    print(f"  CS-HQR 10-Parameter Evaluation — Method: {method}")
    print(f"{'='*60}")
    labels = {
        'P1_Qubits_Required'               : 'P1  Qubits required',
        'P2_Circuit_Depth'                 : 'P2  Circuit depth',
        'P3_Gate_Count'                    : 'P3  Gate count',
        'P4_Encoding_Time_ms'              : 'P4  Encoding time (ms)',
        'P5_Scalability_Factor'            : 'P5  Scalability factor',
        'P6_Information_Preservation_SSIM' : 'P6  SSIM (info. preservation)',
        'P7_Compression_Ratio'             : 'P7  Compression ratio',
        'P8_Memory_Overhead_pct'           : 'P8  Memory overhead (%)',
        'P9_Gate_Complexity_per_Qubit'     : 'P9  Gate complexity / qubit',
        'P10_Implementation_Complexity'    : 'P10 Implementation complexity',
    }
    for key, label in labels.items():
        val = metrics.get(key, 'N/A')
        print(f"  {label:<40} {val}")
    print(f"{'='*60}\n")


def print_comparison_table(all_metrics):
    """Print a summary comparison table for all methods."""
    methods = list(all_metrics.keys())
    cols    = ['P1', 'P2', 'P3', 'P4', 'P6', 'P7']
    col_keys = {
        'P1': 'P1_Qubits_Required',
        'P2': 'P2_Circuit_Depth',
        'P3': 'P3_Gate_Count',
        'P4': 'P4_Encoding_Time_ms',
        'P6': 'P6_Information_Preservation_SSIM',
        'P7': 'P7_Compression_Ratio',
    }
    header = f"{'Method':<12}" + "".join(f"{c:>12}" for c in cols)
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")
    for m in methods:
        row = f"{m:<12}"
        for c in cols:
            val = all_metrics[m].get(col_keys[c], 'N/A')
            row += f"{str(val):>12}"
        print(row)
    print(f"{'='*len(header)}\n")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CS-HQR: Chroma-Subsampled Hybrid Quantum Representation demo.\n"
            "Encodes a color image using quantum circuits and evaluates\n"
            "10-parameter metrics against 11 baseline methods."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to input image (PNG, JPEG, etc.)")
    parser.add_argument("--subsampling", choices=["420","422","444"], default="420",
                        help="Chroma subsampling mode (default: 420)")
    parser.add_argument("--bit-depth", type=int, default=8, dest="bit_depth",
                        help="Color register size in qubits (default: 8)")
    parser.add_argument("--max-size", type=int, default=16, dest="max_size",
                        help="Resize image to at most N x N (default: 16)")
    parser.add_argument("--output", default=None,
                        help="Save reconstructed image to this path")
    parser.add_argument("--output-dir", default=".", dest="output_dir",
                        help="Directory for output files (default: .)")
    parser.add_argument("--channels", action="store_true",
                        help="Save Y, Cb, Cr channel images to output-dir")
    parser.add_argument("--compare", action="store_true",
                        help="Run all 11 baseline methods and print comparison table")
    parser.add_argument("--json", default=None,
                        help="Write evaluation metrics to a JSON file")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\nCS-HQR Demo")
    print("-" * 40)

    img_rgb = load_and_preprocess(args.input, max_size=args.max_size)
    H, W    = img_rgb.shape[:2]
    os.makedirs(args.output_dir, exist_ok=True)

    if args.channels:
        ycbcr_full = rgb_to_ycbcr(img_rgb)
        base = os.path.splitext(os.path.basename(args.input))[0]
        save_channel_images(ycbcr_full, args.output_dir, prefix=base)

    print(f"\nEncoding with CS-HQR  (subsampling=4:{args.subsampling[1]}:{args.subsampling[2]})...")
    result  = encode_cshqr(img_rgb)
    metrics = compute_evaluation_metrics(result, image_size=H*W, original_image=img_rgb)
    print_metrics(metrics)

    recon = decode_cshqr(result)
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(args.output_dir, f"{base}_cshqr_recon.png")
    save_image(recon, out_path)

    all_json_metrics = {'CS-HQR': metrics}

    if args.compare:
        print("Running baseline methods ... (this may take a few minutes)")
        baseline_names = [m for m in METHOD_NAMES if m != 'CS-HQR']
        all_results = encode_image_all_methods(img_rgb, methods=baseline_names)
        for name, res in all_results.items():
            all_json_metrics[name] = compute_evaluation_metrics(res, image_size=H*W)
        all_json_metrics['CS-HQR'] = metrics
        print_comparison_table(all_json_metrics)

    if args.json:
        json_path = args.json
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        json_path = os.path.join(args.output_dir, f"{base}_metrics.json")

    def _jsonify(d):
        if isinstance(d, dict):
            return {k: _jsonify(v) for k, v in d.items()}
        if isinstance(d, np.integer):
            return int(d)
        if isinstance(d, np.floating):
            return float(d)
        return d

    with open(json_path, 'w') as f:
        json.dump(_jsonify(all_json_metrics), f, indent=2)
    print(f"  Metrics JSON: {json_path}\n")


if __name__ == "__main__":
    main()
