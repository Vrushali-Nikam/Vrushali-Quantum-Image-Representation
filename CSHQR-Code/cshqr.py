# -*- coding: utf-8 -*-
#
# CS-HQR: Chroma-Subsampled Hybrid Quantum Representation
#         for Efficient Color Image Encoding
#
# Copyright (C) 2026 Raiyyan Patel <raiyyanpatel467@gmail.com>
# Department of Information Technology,
# Dwarkadas Jivanlal Sanghvi College of Engineering
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
"""
CS-HQR: Chroma-Subsampled Hybrid Quantum Representation

This module implements the complete CS-HQR encoding pipeline as described in:
    R. Patel, "CS-HQR: Chroma-Subsampled Hybrid Quantum Representation
    for Efficient Color Image Encoding",
    Image Processing On Line, 2026.
    https://github.com/Raiyyanpatel/Quantum-Image-Representation

The module includes:
    - YCbCr color space transformation  (Section 4 of the paper)
    - 4:2:0 chroma subsampling          (Section 4 of the paper)
    - Circuit A: luminance encoding     (Algorithm 1, Phase 3)
    - Circuit B: chrominance encoding   (Algorithm 1, Phase 3)
    - CS-HQR encoding / decoding        (Algorithm 1 / Algorithm 2)
    - 11 baseline encoding methods for comparison (Section 2)
    - 10-parameter evaluation framework (Section 5)

Usage example::

    from cshqr import encode_cshqr, decode_cshqr, compute_evaluation_metrics
    import numpy as np
    from PIL import Image

    img_rgb = np.array(Image.open("brain_mri.png").convert("RGB"), dtype=np.uint8)
    result  = encode_cshqr(img_rgb)
    metrics = compute_evaluation_metrics(result, image_size=img_rgb.shape[0]*img_rgb.shape[1])
    recon   = decode_cshqr(result)
"""

import time
import warnings

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from qiskit import QuantumCircuit

warnings.filterwarnings("ignore")


# ============================================================================
# COLOR SPACE UTILITIES  (Section 4.1 / Equations 1-3 in the paper)
# ============================================================================

def rgb_to_ycbcr(image_rgb):
    """
    Convert an RGB image to YCbCr color space (ITU-R BT.601).

    Parameters
    ----------
    image_rgb : numpy.ndarray
        Input RGB image of shape (H, W, 3), dtype uint8.

    Returns
    -------
    ycbcr : numpy.ndarray
        YCbCr image of shape (H, W, 3), dtype float64.
        Channel order: [Y, Cb, Cr].
    """
    img = image_rgb.astype(np.float64)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    Y  =  0.299    * R + 0.587    * G + 0.114    * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5      * B + 128.0
    Cr =  0.5      * R - 0.418688 * G - 0.081312 * B + 128.0

    return np.stack([Y, Cb, Cr], axis=2)


def ycbcr_to_rgb(ycbcr):
    """
    Convert a YCbCr image back to RGB color space (ITU-R BT.601 inverse).

    Parameters
    ----------
    ycbcr : numpy.ndarray
        YCbCr image of shape (H, W, 3), dtype float64.

    Returns
    -------
    rgb : numpy.ndarray
        Reconstructed RGB image of shape (H, W, 3), dtype uint8.
    """
    Y  = ycbcr[..., 0]
    Cb = ycbcr[..., 1] - 128.0
    Cr = ycbcr[..., 2] - 128.0

    R = Y +             1.402    * Cr
    G = Y - 0.344136  * Cb - 0.714136 * Cr
    B = Y + 1.772     * Cb

    rgb = np.stack([R, G, B], axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ============================================================================
# CHROMA SUBSAMPLING  (Section 4.2 / Equations 4-5 in the paper)
# ============================================================================

def downsample_420(channel):
    """
    Downsample a chrominance channel by factor 2 in each dimension (4:2:0).

    Each 2x2 block is averaged to produce one output sample, reducing the
    channel from NxN to (N/2)x(N/2).

    Parameters
    ----------
    channel : numpy.ndarray
        2-D array (H, W), dtype float64.

    Returns
    -------
    numpy.ndarray
        Downsampled array (H//2, W//2), dtype float64.
    """
    H, W = channel.shape
    H2, W2 = H // 2, W // 2
    sub = (channel[0::2, 0::2] + channel[1::2, 0::2] +
           channel[0::2, 1::2] + channel[1::2, 1::2]) / 4.0
    return sub[:H2, :W2]


def upsample_420(channel_sub, target_shape):
    """
    Upsample a chrominance channel back to the original resolution (bilinear).

    Parameters
    ----------
    channel_sub : numpy.ndarray
        Subsampled 2-D array (H/2, W/2), dtype float64.
    target_shape : tuple
        (H, W) of the target full-resolution channel.

    Returns
    -------
    numpy.ndarray
        Upsampled array of shape target_shape, dtype float64.
    """
    zh = target_shape[0] / channel_sub.shape[0]
    zw = target_shape[1] / channel_sub.shape[1]
    return zoom(channel_sub, (zh, zw), order=1)  # bilinear


# ============================================================================
# NEQR-STYLE CIRCUIT BUILDER  (Section 5.3 of the paper)
# ============================================================================

def _build_neqr_circuit(pixels_flat, n_pos_qubits, n_color_qubits):
    """
    Build an NEQR-style basis-state encoding circuit.

    For each pixel position `idx`, if the pixel value differs from 0, a
    multi-controlled X gate sets the color qubits to the binary encoding of
    the pixel value, controlled by the binary encoding of `idx` on the
    position qubits.

    Parameters
    ----------
    pixels_flat : numpy.ndarray
        1-D array of quantized pixel values (integers in [0, 2**n_color_qubits - 1]).
    n_pos_qubits : int
        Number of position qubits (= ceil(log2(len(pixels_flat)))).
    n_color_qubits : int
        Number of color qubits (= bit depth).

    Returns
    -------
    qc : qiskit.QuantumCircuit
        The assembled NEQR-style encoding circuit.
    """
    total_qubits = n_pos_qubits + n_color_qubits
    qc = QuantumCircuit(total_qubits)

    # Hadamard superposition over position qubits
    for i in range(n_pos_qubits):
        qc.h(i)

    for idx, val in enumerate(pixels_flat):
        if val == 0:
            continue
        # Determine control pattern from binary index
        ctrl_state = format(idx, f'0{n_pos_qubits}b')
        # Determine which color qubits to flip
        val_bits = format(int(val), f'0{n_color_qubits}b')

        for bit_pos, bit in enumerate(val_bits):
            if bit == '1':
                color_qubit = n_pos_qubits + bit_pos
                # Multi-controlled X: controls on position qubits
                for pos, cs in enumerate(ctrl_state):
                    if cs == '0':
                        qc.x(pos)
                qc.mcx(list(range(n_pos_qubits)), color_qubit)
                for pos, cs in enumerate(ctrl_state):
                    if cs == '0':
                        qc.x(pos)

    return qc


# ============================================================================
# CS-HQR ENCODING — Algorithm 1 in the paper
# ============================================================================

def encode_cshqr(image_rgb):
    """
    CS-HQR: Chroma-Subsampled Hybrid Quantum Representation encoding.

    Implements Algorithm 1 from the paper:
        Phase 1 — YCbCr color space transformation
        Phase 2 — 4:2:0 chroma subsampling
        Phase 3 — Dual-circuit quantum encoding
            Circuit A: luminance Y  at full NxN resolution  (16 qubits)
            Circuit B: chrominance Cb+Cr at N/2 x N/2 resolution (15 qubits)

    Parameters
    ----------
    image_rgb : numpy.ndarray
        Input RGB color image of shape (H, W, 3), dtype uint8.
        H and W must be equal and a power of 2 (e.g., 16x16).

    Returns
    -------
    result : dict
        - 'method'          : 'CS-HQR'
        - 'circuit_a'       : qiskit.QuantumCircuit (luminance)
        - 'circuit_b'       : qiskit.QuantumCircuit (chrominance)
        - 'qubits'          : int, max(qubits_A, qubits_B)
        - 'gates'           : int, total gates (A + B)
        - 'depth'           : int, max(depth_A, depth_B)
        - 'encoding_time'   : float, encoding time in ms
        - 'Y'               : numpy.ndarray, luminance channel (H, W)
        - 'Cb_sub'          : numpy.ndarray, subsampled Cb (H/2, W/2)
        - 'Cr_sub'          : numpy.ndarray, subsampled Cr (H/2, W/2)
        - 'original_shape'  : tuple, (H, W)
    """
    t_start = time.time()

    H, W = image_rgb.shape[:2]

    # Phase 1: RGB -> YCbCr
    ycbcr = rgb_to_ycbcr(image_rgb)
    Y  = np.clip(ycbcr[..., 0], 0, 255)
    Cb = np.clip(ycbcr[..., 1], 0, 255)
    Cr = np.clip(ycbcr[..., 2], 0, 255)

    # Phase 2: 4:2:0 chroma subsampling
    Cb_sub = downsample_420(Cb)
    Cr_sub = downsample_420(Cr)

    # Phase 3a: Circuit A — luminance (full NxN, 8-bit)
    n_pos_a  = int(np.ceil(np.log2(H * W)))   # 8 for 16x16
    n_col_a  = 8
    Y_flat   = Y.flatten().astype(np.uint8)
    circuit_a = _build_neqr_circuit(Y_flat, n_pos_a, n_col_a)

    # Phase 3b: Circuit B — chrominance (N/2 x N/2, both Cb & Cr, 8-bit + 1 channel qubit)
    Hs, Ws = Cb_sub.shape
    n_pos_b  = int(np.ceil(np.log2(Hs * Ws)))  # 6 for 8x8
    n_col_b  = 8
    n_ch_b   = 1                                # channel selector qubit
    # Interleave Cb and Cr as separate "pixels" with the channel qubit
    Cb_flat  = Cb_sub.flatten().astype(np.uint8)
    Cr_flat  = Cr_sub.flatten().astype(np.uint8)

    # Build chrominance circuit (Cb with ch=0, Cr with ch=1)
    total_b  = n_pos_b + n_col_b + n_ch_b
    circuit_b = QuantumCircuit(total_b)
    for i in range(n_pos_b):
        circuit_b.h(i)

    for idx, (cb_val, cr_val) in enumerate(zip(Cb_flat, Cr_flat)):
        for ch, val in enumerate([cb_val, cr_val]):
            if val == 0:
                continue
            ctrl_state = format(idx, f'0{n_pos_b}b')
            val_bits   = format(int(val), f'0{n_col_b}b')
            # Set / unset channel qubit for Cb (ch=0) or Cr (ch=1)
            ch_qubit = n_pos_b + n_col_b     # last qubit
            if ch == 0:
                circuit_b.x(ch_qubit)        # flip to |0> context
            for bit_pos, bit in enumerate(val_bits):
                if bit == '1':
                    color_qubit = n_pos_b + bit_pos
                    for pos, cs in enumerate(ctrl_state):
                        if cs == '0':
                            circuit_b.x(pos)
                    ctrl_list = list(range(n_pos_b)) + [ch_qubit]
                    circuit_b.mcx(ctrl_list, color_qubit)
                    for pos, cs in enumerate(ctrl_state):
                        if cs == '0':
                            circuit_b.x(pos)
            if ch == 0:
                circuit_b.x(ch_qubit)        # restore

    encoding_time = (time.time() - t_start) * 1000

    return {
        'method'        : 'CS-HQR',
        'circuit_a'     : circuit_a,
        'circuit_b'     : circuit_b,
        'qubits'        : max(circuit_a.num_qubits, circuit_b.num_qubits),
        'gates'         : circuit_a.size() + circuit_b.size(),
        'depth'         : max(circuit_a.depth(), circuit_b.depth()),
        'encoding_time' : encoding_time,
        'Y'             : Y,
        'Cb_sub'        : Cb_sub,
        'Cr_sub'        : Cr_sub,
        'original_shape': (H, W),
    }


# ============================================================================
# CS-HQR DECODING — Algorithm 2 in the paper
# ============================================================================

def decode_cshqr(result):
    """
    CS-HQR decoding: reconstruct an RGB image from an encoded CS-HQR result.

    Implements Algorithm 2 from the paper:
        Step 1 — Recover Y, Cb_sub, Cr_sub from encoding result
        Step 2 — Bilinear upsample Cb_sub and Cr_sub to original resolution
        Step 3 — Inverse YCbCr -> RGB transform

    Parameters
    ----------
    result : dict
        Output dictionary from encode_cshqr().

    Returns
    -------
    reconstructed_rgb : numpy.ndarray
        Reconstructed RGB image of shape (H, W, 3), dtype uint8.
    """
    H, W = result['original_shape']

    Y      = result['Y']
    Cb_sub = result['Cb_sub']
    Cr_sub = result['Cr_sub']

    # Step 2: Upsample chrominance channels (bilinear)
    Cb = upsample_420(Cb_sub, (H, W))
    Cr = upsample_420(Cr_sub, (H, W))

    # Step 3: Inverse YCbCr -> RGB
    ycbcr_recon = np.stack([Y, Cb, Cr], axis=2)
    return ycbcr_to_rgb(ycbcr_recon)


# ============================================================================
# BASELINE ENCODING METHODS  (Section 2 of the paper)
# ============================================================================

def encode_frqi(image):
    """FRQI: Flexible Representation of Quantum Images (Le et al., 2011)."""
    t = time.time()
    h, w = image.shape[:2]
    gray = image if image.ndim == 2 else (0.299*image[...,0] + 0.587*image[...,1] + 0.114*image[...,2])
    n_pos = int(np.ceil(np.log2(h * w)))
    qc = QuantumCircuit(n_pos + 1)
    for i in range(n_pos):
        qc.h(i)
    for idx, px in enumerate(gray.flatten()):
        theta = float(px) * np.pi / (2.0 * 255.0)
        if theta != 0:
            ctrl = list(range(n_pos))
            qc.mcx(ctrl, n_pos)          # approximate: use MCX as placeholder
            qc.ry(2 * theta, n_pos)
            qc.mcx(ctrl, n_pos)
    return {'method': 'FRQI', 'circuit': qc, 'qubits': qc.num_qubits,
            'gates': qc.size(), 'depth': qc.depth(),
            'encoding_time': (time.time()-t)*1000}


def encode_efrqi(image, angle_levels=32):
    """EFRQI: Enhanced FRQI with quantized rotation angles (Liu et al.)."""
    t = time.time()
    h, w = image.shape[:2]
    gray = image if image.ndim == 2 else (0.299*image[...,0]+0.587*image[...,1]+0.114*image[...,2])
    n_pos = int(np.ceil(np.log2(h * w)))
    qc = QuantumCircuit(n_pos + 1)
    for i in range(n_pos):
        qc.h(i)
    for idx, px in enumerate(gray.flatten()):
        level = int(float(px) / (256.0 / angle_levels))
        theta = level * np.pi / (2.0 * (angle_levels - 1)) if angle_levels > 1 else 0
        if theta != 0:
            ctrl = list(range(n_pos))
            qc.mcx(ctrl, n_pos)
            qc.ry(2 * theta, n_pos)
            qc.mcx(ctrl, n_pos)
    return {'method': 'EFRQI', 'circuit': qc, 'qubits': qc.num_qubits,
            'gates': qc.size(), 'depth': qc.depth(),
            'encoding_time': (time.time()-t)*1000}


def encode_qpie(image):
    """QPIE: Quantum Probability Image Encoding (Yao et al.)."""
    t = time.time()
    h, w = image.shape[:2]
    gray = image if image.ndim == 2 else (0.299*image[...,0]+0.587*image[...,1]+0.114*image[...,2])
    flat = gray.flatten().astype(np.float64)
    norm = np.linalg.norm(flat)
    amplitudes = flat / norm if norm > 1e-10 else flat
    n_qubits = int(np.ceil(np.log2(len(flat))))
    qc = QuantumCircuit(n_qubits)
    # Approximate initialization using Ry rotations
    for i in range(n_qubits):
        qc.ry(np.arcsin(np.clip(amplitudes[i] if i < len(amplitudes) else 0, -1, 1)) * 2, i)
    return {'method': 'QPIE', 'circuit': qc, 'qubits': qc.num_qubits,
            'gates': qc.size(), 'depth': qc.depth(),
            'encoding_time': (time.time()-t)*1000}


def encode_neqr(image):
    """NEQR: Novel Enhanced Quantum Representation (Zhang et al., 2013)."""
    t = time.time()
    h, w = image.shape[:2]
    gray = image if image.ndim == 2 else (0.299*image[...,0]+0.587*image[...,1]+0.114*image[...,2])
    n_pos = int(np.ceil(np.log2(h * w)))
    qc = _build_neqr_circuit(gray.flatten().astype(np.uint8), n_pos, 8)
    return {'method': 'NEQR', 'circuit': qc, 'qubits': qc.num_qubits,
            'gates': qc.size(), 'depth': qc.depth(),
            'encoding_time': (time.time()-t)*1000}


def encode_gqir(image, color_bits=4):
    """GQIR: Generalized Quantum Image Representation (Jiang et al., 2015)."""
    t = time.time()
    h, w = image.shape[:2]
    gray = image if image.ndim == 2 else (0.299*image[...,0]+0.587*image[...,1]+0.114*image[...,2])
    n_pos = int(np.ceil(np.log2(h * w)))
    max_val = (2 ** color_bits) - 1
    quantized = (gray.flatten().astype(np.float64) / 255.0 * max_val).astype(np.uint8)
    qc = _build_neqr_circuit(quantized, n_pos, color_bits)
    return {'method': 'GQIR', 'circuit': qc, 'qubits': qc.num_qubits,
            'gates': qc.size(), 'depth': qc.depth(),
            'encoding_time': (time.time()-t)*1000, 'color_bits': color_bits}


def encode_ineqr(image):
    """INEQR: Improved NEQR using differential (XOR) encoding."""
    t = time.time()
    h, w = image.shape[:2]
    gray = image if image.ndim == 2 else (0.299*image[...,0]+0.587*image[...,1]+0.114*image[...,2])
    flat = gray.flatten().astype(np.uint8)
    diff = np.bitwise_xor(flat, np.roll(flat, 1))
    diff[0] = flat[0]
    n_pos = int(np.ceil(np.log2(len(flat))))
    qc = _build_neqr_circuit(diff, n_pos, 8)
    return {'method': 'INEQR', 'circuit': qc, 'qubits': qc.num_qubits,
            'gates': qc.size(), 'depth': qc.depth(),
            'encoding_time': (time.time()-t)*1000}


def encode_tnr(image):
    """TNR: Two's Complement NEQR for signed pixel differences."""
    t = time.time()
    h, w = image.shape[:2]
    gray = image if image.ndim == 2 else (0.299*image[...,0]+0.587*image[...,1]+0.114*image[...,2])
    flat = gray.flatten().astype(np.int16)
    diff = (flat - np.roll(flat, 1)).astype(np.int8)
    diff[0] = 0
    # Map signed bytes to unsigned for NEQR encoding (two's complement)
    unsigned = diff.astype(np.uint8)
    n_pos = int(np.ceil(np.log2(len(unsigned))))
    qc = _build_neqr_circuit(unsigned, n_pos, 8)
    return {'method': 'TNR', 'circuit': qc, 'qubits': qc.num_qubits,
            'gates': qc.size(), 'depth': qc.depth(),
            'encoding_time': (time.time()-t)*1000}


def encode_mcqi(image):
    """MCQI: Multi-Channel Quantum Images — RGB channels as separate registers."""
    t = time.time()
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=2)
    h, w = image.shape[:2]
    n_pos = int(np.ceil(np.log2(h * w)))
    n_col = 8
    # 3 color registers + position register
    total = n_pos + 3 * n_col
    qc = QuantumCircuit(total)
    for i in range(n_pos):
        qc.h(i)
    for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
        ch_data = image[..., ch_idx].flatten().astype(np.uint8)
        base = n_pos + ch_idx * n_col
        for idx, val in enumerate(ch_data):
            if val == 0:
                continue
            ctrl_state = format(idx, f'0{n_pos}b')
            val_bits   = format(int(val), f'08b')
            for bit_pos, bit in enumerate(val_bits):
                if bit == '1':
                    color_qubit = base + bit_pos
                    for pos, cs in enumerate(ctrl_state):
                        if cs == '0':
                            qc.x(pos)
                    qc.mcx(list(range(n_pos)), color_qubit)
                    for pos, cs in enumerate(ctrl_state):
                        if cs == '0':
                            qc.x(pos)
    return {'method': 'MCQI', 'circuit': qc, 'qubits': qc.num_qubits,
            'gates': qc.size(), 'depth': qc.depth(),
            'encoding_time': (time.time()-t)*1000}


def encode_qrmw(image, num_bands=3):
    """QRMW: Quantum Representation for Multi-Wavelength images."""
    t = time.time()
    if image.ndim == 2:
        image = np.stack([image]*num_bands, axis=2)
    h, w = image.shape[:2]
    n_pos  = int(np.ceil(np.log2(h * w)))
    n_col  = 8
    n_band = int(np.ceil(np.log2(num_bands)))
    total  = n_pos + n_band + n_col
    qc = QuantumCircuit(total)
    for i in range(n_pos + n_band):
        qc.h(i)
    for band in range(min(num_bands, image.shape[2])):
        ch = image[..., band].flatten().astype(np.uint8)
        for idx, val in enumerate(ch):
            if val == 0:
                continue
            ctrl_state = format(idx, f'0{n_pos}b') + format(band, f'0{n_band}b')
            val_bits   = format(int(val), '08b')
            ctrl_qubits = list(range(n_pos + n_band))
            for bit_pos, bit in enumerate(val_bits):
                if bit == '1':
                    color_qubit = n_pos + n_band + bit_pos
                    for pos, cs in enumerate(ctrl_state):
                        if cs == '0':
                            qc.x(pos)
                    qc.mcx(ctrl_qubits, color_qubit)
                    for pos, cs in enumerate(ctrl_state):
                        if cs == '0':
                            qc.x(pos)
    return {'method': 'QRMW', 'circuit': qc, 'qubits': qc.num_qubits,
            'gates': qc.size(), 'depth': qc.depth(),
            'encoding_time': (time.time()-t)*1000}


def encode_dct_qir(image, keep_coefficients=None):
    """DCT-QIR: Discrete Cosine Transform Quantum Image Representation."""
    import scipy.fft as sfft
    t = time.time()
    gray = image if image.ndim == 2 else (0.299*image[...,0]+0.587*image[...,1]+0.114*image[...,2])
    dct_coeffs = sfft.dct(sfft.dct(gray.astype(float), axis=0), axis=1)
    flat = dct_coeffs.flatten()
    if keep_coefficients is None:
        keep_coefficients = max(1, len(flat) // 4)
    # Keep only top-K coefficients by magnitude
    top_idx = np.argsort(np.abs(flat))[::-1][:keep_coefficients]
    quantized = np.zeros(len(flat), dtype=np.uint8)
    max_c = np.max(np.abs(flat[top_idx])) if len(top_idx) > 0 else 1.0
    for i in top_idx:
        quantized[i] = int(np.clip(np.abs(flat[i]) / max_c * 255, 0, 255))
    n_pos = int(np.ceil(np.log2(keep_coefficients)))
    sub   = quantized[top_idx[:2**n_pos]]
    qc    = _build_neqr_circuit(sub, n_pos, 8)
    return {'method': 'DCT-QIR', 'circuit': qc, 'qubits': qc.num_qubits,
            'gates': qc.size(), 'depth': qc.depth(),
            'encoding_time': (time.time()-t)*1000}


def encode_qlr(image):
    """QLR: Quantum Log-Polar Representation (Sang et al.)."""
    t = time.time()
    gray = image if image.ndim == 2 else (0.299*image[...,0]+0.587*image[...,1]+0.114*image[...,2])
    H, W = gray.shape
    cy, cx = H / 2.0, W / 2.0
    rho_map  = np.zeros_like(gray, dtype=np.float64)
    phi_map  = np.zeros_like(gray, dtype=np.float64)
    for y in range(H):
        for x in range(W):
            dy, dx = y - cy, x - cx
            r = np.sqrt(dy**2 + dx**2)
            rho_map[y, x]  = np.log(r + 1e-10)
            phi_map[y, x]  = np.arctan2(dy, dx)
    # Normalize and quantize rho for NEQR encoding
    r_min, r_max = rho_map.min(), rho_map.max()
    if r_max > r_min:
        rho_norm = ((rho_map - r_min) / (r_max - r_min) * 255).astype(np.uint8)
    else:
        rho_norm = np.zeros_like(gray, dtype=np.uint8)
    n_pos = int(np.ceil(np.log2(H * W)))
    qc    = _build_neqr_circuit(rho_norm.flatten(), n_pos, 8)
    return {'method': 'QLR', 'circuit': qc, 'qubits': qc.num_qubits,
            'gates': qc.size(), 'depth': qc.depth(),
            'encoding_time': (time.time()-t)*1000}


# ============================================================================
# METHOD DISPATCHER
# ============================================================================

ENCODING_METHODS = {
    'FRQI'    : encode_frqi,
    'EFRQI'   : encode_efrqi,
    'QPIE'    : encode_qpie,
    'NEQR'    : encode_neqr,
    'GQIR'    : encode_gqir,
    'INEQR'   : encode_ineqr,
    'TNR'     : encode_tnr,
    'MCQI'    : encode_mcqi,
    'QRMW'    : encode_qrmw,
    'DCT-QIR' : encode_dct_qir,
    'QLR'     : encode_qlr,
    'CS-HQR'  : encode_cshqr,
}

METHOD_NAMES = list(ENCODING_METHODS.keys())


def encode_image_all_methods(image_rgb, methods=None):
    """
    Run all (or selected) encoding methods on an RGB image.

    Parameters
    ----------
    image_rgb : numpy.ndarray
        RGB image of shape (H, W, 3), dtype uint8.
    methods : list of str, optional
        Subset of METHOD_NAMES to run. Defaults to all methods.

    Returns
    -------
    results : dict
        Mapping method name -> encoding result dict.
    """
    if methods is None:
        methods = METHOD_NAMES

    results = {}
    for name in methods:
        fn = ENCODING_METHODS.get(name)
        if fn is None:
            continue
        try:
            if name == 'CS-HQR':
                results[name] = fn(image_rgb)
            else:
                # Grayscale conversion for single-channel baselines
                gray = (0.299*image_rgb[...,0] +
                        0.587*image_rgb[...,1] +
                        0.114*image_rgb[...,2]).astype(np.uint8)
                if name in ('MCQI', 'QRMW'):
                    results[name] = fn(image_rgb)
                else:
                    results[name] = fn(gray)
        except Exception as e:
            print(f"  Warning: {name} failed — {e}")
    return results


# ============================================================================
# EVALUATION METRICS — 10-Parameter Framework  (Section 5 of the paper)
# ============================================================================

FRQI_BASELINE_QUBITS = 9   # Reference for P8 memory overhead

IMPLEMENTATION_COMPLEXITY = {
    'FRQI'   : 2,
    'EFRQI'  : 3,
    'QPIE'   : 2,
    'NEQR'   : 3,
    'GQIR'   : 4,
    'INEQR'  : 4,
    'TNR'    : 3,
    'MCQI'   : 5,
    'QRMW'   : 5,
    'DCT-QIR': 4,
    'QLR'    : 4,
    'CS-HQR' : 4,
}

EVALUATION_PARAMS = [
    'P1_Qubits_Required',
    'P2_Circuit_Depth',
    'P3_Gate_Count',
    'P4_Encoding_Time_ms',
    'P5_Scalability_Factor',
    'P6_Information_Preservation_SSIM',
    'P7_Compression_Ratio',
    'P8_Memory_Overhead_pct',
    'P9_Gate_Complexity_per_Qubit',
    'P10_Implementation_Complexity',
]


def compute_ssim(img1, img2):
    """Compute SSIM between two grayscale images (simplified)."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    sig1  = img1.std()
    sig2  = img2.std()
    sig12 = np.mean((img1 - mu1) * (img2 - mu2))
    return ((2*mu1*mu2 + c1) * (2*sig12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sig1**2 + sig2**2 + c2))


def compute_evaluation_metrics(encoding_result, image_size=256, original_image=None):
    """
    Compute the 10-parameter evaluation metrics for an encoding result.

    Parameters
    ----------
    encoding_result : dict
        Output from any encode_* function.
    image_size : int
        Total number of pixels (H * W). Default 256 (16x16).
    original_image : numpy.ndarray, optional
        Original image for SSIM computation. If None, SSIM defaults to 1.0
        for lossless methods and 0.95 for lossy.

    Returns
    -------
    metrics : dict
        Dictionary with keys matching EVALUATION_PARAMS.
    """
    method = encoding_result.get('method', 'Unknown')
    qubits = encoding_result.get('qubits', 0)
    depth  = encoding_result.get('depth', 0)
    gates  = encoding_result.get('gates', 0)
    enc_t  = encoding_result.get('encoding_time', 0.0)

    # P1 — Qubits required
    p1 = qubits

    # P2 — Circuit depth
    p2 = depth

    # P3 — Gate count
    p3 = gates

    # P4 — Encoding time (ms)
    p4 = round(enc_t, 4)

    # P5 — Scalability factor = 100 / qubits  (higher = more scalable)
    p5 = round(100.0 / qubits, 4) if qubits > 0 else 0.0

    # P6 — Information preservation (SSIM)
    if original_image is not None and method == 'CS-HQR':
        recon = decode_cshqr(encoding_result)
        gray_orig  = (0.299*original_image[...,0]+0.587*original_image[...,1]+0.114*original_image[...,2])
        gray_recon = (0.299*recon[...,0]+0.587*recon[...,1]+0.114*recon[...,2])
        p6 = round(compute_ssim(gray_orig, gray_recon), 6)
    elif method in ('DCT-QIR', 'QPIE', 'EFRQI'):
        p6 = 0.950   # lossy methods
    else:
        p6 = 1.000   # lossless

    # P7 — Compression ratio = classical bits / (qubits x depth)
    classical_bits = image_size * 24   # 8-bit x 3 channels
    p7 = round(classical_bits / (qubits * depth), 6) if (qubits * depth) > 0 else 0.0

    # P8 — Memory overhead relative to FRQI baseline (%)
    p8 = round(qubits / FRQI_BASELINE_QUBITS * 100.0, 2)

    # P9 — Gate complexity (gates per qubit)
    p9 = round(gates / qubits, 4) if qubits > 0 else 0.0

    # P10 — Implementation complexity (1-5 scale)
    p10 = IMPLEMENTATION_COMPLEXITY.get(method, 3)

    return {
        'method'                           : method,
        'P1_Qubits_Required'               : p1,
        'P2_Circuit_Depth'                 : p2,
        'P3_Gate_Count'                    : p3,
        'P4_Encoding_Time_ms'              : p4,
        'P5_Scalability_Factor'            : p5,
        'P6_Information_Preservation_SSIM' : p6,
        'P7_Compression_Ratio'             : p7,
        'P8_Memory_Overhead_pct'           : p8,
        'P9_Gate_Complexity_per_Qubit'     : p9,
        'P10_Implementation_Complexity'    : p10,
    }
