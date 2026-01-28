# CS-HQR: Chroma-Subsampled Hybrid Quantum Representation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Latest-purple)](https://qiskit.org/)

## Overview

**CS-HQR (Chroma Subsampling Hybrid Quantum Representation)** is a novel quantum image encoding methodology that bridges classical image compression techniques with quantum circuit optimization. This repository contains the research implementation submitted to **Springer Neural Processing Letters**.

### Key Innovation

CS-HQR introduces a breakthrough approach to quantum image representation by:
- Leveraging **YCbCr color space transformation**
- Implementing **4:2:0 chroma subsampling** (inspired by JPEG/MPEG standards)
- Utilizing **asymmetric quantum circuits** for luminance and chrominance channels
- Achieving **51% gate count reduction** compared to existing methods while maintaining **SSIM > 0.98**

## Features

- ✅ **Efficient Quantum Encoding**: 50% reduction in encoding complexity
- ✅ **Perceptual Fidelity**: SSIM consistently exceeding 0.9999
- ✅ **Scalability**: O(n·2^n) complexity vs O(2^(2n)) for amplitude-based methods
- ✅ **NISQ-Ready**: Optimized for Near-Term Intermediate-Scale Quantum devices
- ✅ **Comprehensive Comparison**: Benchmarked against 10 state-of-the-art methods
- ✅ **Medical Imaging Dataset**: Validated on 6,097 brain scan slices

## Methods Compared

The project compares CS-HQR against 10 state-of-the-art quantum image representation methods:

1. **FRQI** - Flexible Representation of Quantum Images
2. **NEQR** - Novel Enhanced Quantum Representation
3. **GQIR** - Generalized Quantum Image Representation
4. **MCQI** - Multi-Channel Quantum Images
5. **QRMW** - Quantum Representation for Multi-Wavelength Images
6. **EFRQI** - Enhanced Flexible Representation of Quantum Images
7. **2D-QSNA** - 2D Quantum State Normalization Approach
8. **INEQR** - Improved Novel Enhanced Quantum Representation
9. **QPIE** - Quantum Probability Image Encoding
10. **QLR** - Quantum Log-polar Representation
11. **CS-HQR** - Our proposed method

## Architecture

### Dual-Circuit Design
- **Luminance Circuit**: Full-resolution encoding (16 qubits)
- **Chrominance Circuit**: Quarter-resolution encoding (15 qubits)
- **Parallel Processing**: Circuits can run simultaneously

### Technical Specifications
- **Color Space**: YCbCr (exploiting human visual perception)
- **Subsampling Ratio**: 4:2:0 (75% fewer chroma pixels)
- **Gate Reduction**: ~51% compared to MCQI
- **Data Reduction**: 75% chrominance data

## Repository Structure

```
quantum image/
├── FINAL_CSHQR.ipynb                    # Main implementation notebook
├── CSHQR_Research_Paper.tex             # LaTeX source for the research paper
├── CSHQR_Research_Paper.pdf             # Compiled research paper
├── sn-jnl.cls                           # Springer journal class file
├── cuted.sty                            # LaTeX style file
│
├── 2D/                                  # Medical imaging dataset (MINC format)
│   ├── 2u.2dus.00001sm.mnc
│   ├── 2u.2dus.00002sm.mnc
│   └── ... (6,097 images total)
│
├── CSHQR_Results/                       # Output directory
│   ├── checkpoints/                     # Progress checkpoints (every 500 images)
│   ├── figures/                         # Generated plots and charts
│   └── tables/                          # Evaluation results tables
│
├── experiment_summary.csv               # Summary of all experiments
├── final_10_parameter_evaluation.csv    # 10-parameter evaluation results
├── method_comparison_results.csv        # Comparative analysis results
├── full_comparison_results.csv          # Detailed comparison data
│
├── comparison_charts.png                # Visual comparison charts
├── radar_chart.png                      # Multi-parameter radar chart
├── cshqr_analysis.png                   # CS-HQR specific analysis
└── sys_arch_final.png                   # System architecture diagram
```

## Requirements

### Core Dependencies
```
Python >= 3.8
qiskit >= 0.45.0
numpy >= 1.21.0
pandas >= 1.3.0
scipy >= 1.7.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### Image Processing
```
opencv-python >= 4.5.0
Pillow >= 8.0.0
nibabel >= 3.2.0
scikit-image >= 0.18.0
```

### Additional Tools
```
tqdm >= 4.62.0
jupyter >= 1.0.0
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/cshqr-project.git
   cd cshqr-project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook FINAL_CSHQR-VSN.ipynb
   ```

## Usage

### Quick Start

1. **Open the main notebook**: `FINAL_CSHQR-VSN.ipynb`
2. **Configure paths**: Update the dataset path in the configuration cell
3. **Run all cells**: Execute the notebook sequentially

### Dataset Configuration

The notebook automatically detects the dataset location. If needed, update the path:

```python
DATASET_PATH = r"C:\path\to\your\dataset\group4"
```

### Checkpointing

Results are automatically saved every 500 images to prevent data loss:
- Checkpoints stored in: `CSHQR_Results/checkpoints/`
- Resume from checkpoint by loading the latest `.pkl` file

## Evaluation Metrics

The project evaluates methods across **10 comprehensive parameters**:

1. **P1**: Qubits Required
2. **P2**: Circuit Depth
3. **P3**: Gate Count
4. **P4**: Encoding Time (ms)
5. **P5**: Scalability Factor
6. **P6**: Information Loss
7. **P7**: Compression Ratio
8. **P8**: Memory Overhead
9. **P9**: Gate Complexity
10. **P10**: Implementation Complexity

### Image Quality Metrics
- **SSIM** (Structural Similarity Index)
- **PSNR** (Peak Signal-to-Noise Ratio)
- **MSE** (Mean Squared Error)

## Results

### Key Findings

| Metric | CS-HQR | MCQI | Improvement |
|--------|--------|------|-------------|
| Gate Count | ~49% | 100% | **51% reduction** |
| Qubits (Luma) | 16 | 18 | 2 qubits saved |
| Qubits (Chroma) | 15 | 18 | 3 qubits saved |
| SSIM | >0.9999 | 1.0 | Minimal loss |
| Chroma Data | 25% | 100% | **75% reduction** |

### Performance Highlights
- ✅ **51% gate count reduction** compared to MCQI
- ✅ **SSIM > 0.98** maintained across all test images
- ✅ **O(n·2^n)** scalability vs O(2^(2n)) for amplitude-based methods
- ✅ Validated on **6,097 medical imaging slices**

## Research Paper

The complete research paper is available in this repository:
- **LaTeX Source**: [VSNIKAM-Final.tex](VSNIKAM-Final.tex)
- **PDF**: [VSNIKAM_P1_FINAL.pdf](VSNIKAM_P1_FINAL.pdf)
- **Target Journal**: Springer Neural Processing Letters
- **Status**: Submitted (January 2026)



## Visualization

The project generates publication-quality visualizations:

### Comparison Charts
- Multi-method performance comparison
- Gate count analysis
- Circuit depth comparison
- Scalability analysis

### Radar Charts
- 10-parameter evaluation visualization
- Method-by-method comparison
- CS-HQR performance profile

### System Architecture
- Dual-circuit design diagram
- YCbCr transformation pipeline
- Subsampling scheme visualization

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

**Vrushali Nikam** et al.
- Email: vrushali.nik@gmail.com 
- Institution: Department of Computer Engineering

## Acknowledgments

- **Dataset**: Medical imaging dataset (MINC format) - 6,097 brain scan slices
- **Quantum Framework**: IBM Qiskit
- **Target Journal**: Springer Neural Processing Letters
- **Inspiration**: Classical image compression (JPEG/MPEG standards)

## Related Projects

- [SAHQR](FINAL_CSHQR-VSN.ipynb) - Spatial Attention Hybrid Quantum Representation
- Other quantum image representation implementations

## Support

For questions, issues, or collaboration:
- 📧 Email: vrushali.nik@gmail.com 
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/cshqr-project/issues)
- 📖 Documentation: See research paper and notebook comments

## Project Status

- ✅ Implementation Complete
- ✅ Experimental Validation Complete (6,097 images)
- ✅ Research Paper Submitted
- 🔄 Under Review (Springer Neural Processing Letters)

---

**Date**: January 2026  
**Version**: 1.0.0  
**Target**: NISQ Devices and Near-Term Quantum Computing Applications

