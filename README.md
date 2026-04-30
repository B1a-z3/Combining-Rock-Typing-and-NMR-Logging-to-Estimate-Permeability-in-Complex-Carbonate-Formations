# Combining Rock Typing and NMR Logging to Estimate Permeability in Complex Carbonate Formations

**Course:** DASC 5391 – Data Science Applications  
**Author:** Nihal Malavalli Lokesh | Student ID: 1002200461  
**Institution:** Kansas State University

---

## Overview

This project develops a **rock-typing-guided SDR (Schlumberger-Doll Research) permeability calibration workflow** for the Arbuckle carbonate aquifer in Kansas. Instead of fitting a single global SDR model to all data, the workflow first classifies reservoir rock into distinct **hydraulic flow units (HFUs)** using K-means clustering on the Flow Zone Indicator (FZI), then calibrates the SDR model independently for each rock type.

The result: prediction error (RMSE on log₁₀ scale) drops from **1.212** (global model) to as low as **0.374** (Cluster 1), demonstrating the value of rock typing as a precondition for NMR-based permeability estimation.

---

## Repository Structure

```
.
├── Research.ipynb          # Main analysis notebook (full pipeline)
├── log_arbuckle.csv        # NMR well log data — Wellington KGS 1-32 (1,997 rows, 32 cols)
├── log_cutter.csv          # NMR well log data — Cutter KGS-1 (5,173 rows, 26 cols)
├── arbucklecoreperm.csv    # Core plug measurements — Arbuckle well (405 samples)
├── cuttercoreperm.csv      # Core plug measurements — Cutter well (289 samples)
└── README.md               # This file
```

---

## Data Description

### Well Log Files (`log_arbuckle.csv`, `log_cutter.csv`)
NMR well log measurements sampled at half-foot depth intervals.

| Column | Description |
|--------|-------------|
| `DEPT.F` | Depth (ft) |
| `4ms` … `2048ms` | NMR T₂ amplitude distribution across 10 bins (4 ms to 2,048 ms) |
| `PHItmri` | NMR-derived total porosity |
| Other columns | Standard wireline logs (GR, NPHI, density, etc.) |

### Core Plug Files (`arbucklecoreperm.csv`, `cuttercoreperm.csv`)
Laboratory measurements on core plugs.

| Column | Description |
|--------|-------------|
| `D_from_ft`, `D_to_ft` | Depth interval of the core plug (ft) |
| `k_max` | Maximum horizontal permeability (mD) |
| `k_90`, `k_vert` | 90° and vertical permeability (mD) |
| `Poro_percent` | Core porosity (%) |
| `G_den_G_CC` | Grain density (g/cc) |

---

## Workflow

The full pipeline is implemented in `Research.ipynb` and follows these stages:

### 1. Data Loading & Preprocessing
- Load NMR log and core plug CSV files for both wells
- Standardize column names, units, and numeric types
- Filter rows with invalid or missing values

### 2. Depth Matching
- Average all NMR log measurements within each core plug's depth interval `[D_from_ft, D_to_ft]`
- Enforce a one-to-one mapping between each core plug and its closest valid log interval
- Removes ambiguous or duplicate matches
- **Output:** 472 matched samples (385 Arbuckle + 87 Cutter)

### 3. Feature Engineering
The following petrophysical descriptors are computed:

| Feature | Formula | Description |
|---------|---------|-------------|
| `phi_core` | `Poro_percent / 100` | Fractional core porosity |
| `RQI` | `0.0314 × √(k / φ)` | Reservoir Quality Index (μm) |
| `phi_z` | `φ / (1 − φ)` | Normalized porosity (void ratio) |
| `FZI` | `RQI / phi_z` | Flow Zone Indicator (μm) |
| `T2gm` | `exp[Σ(wᵢ ln Tᵢ) / Σwᵢ]` | Geometric mean T₂ from NMR bins (ms) |

Near-zero safeguards (`EPS = 1e-12`) are applied throughout to prevent undefined log values.

### 4. Rock Typing — K-means on log(FZI)
- Apply K-means clustering to `log(FZI)` values
- Determine optimal cluster count using the **Elbow method** (WSS for k = 1–15)
- **k = 8** selected as the optimal number of hydraulic flow units
- Run with `n_init = 25` random restarts to ensure stability
- Re-order cluster labels by ascending mean FZI:
  - **Cluster 1** = tightest rock (lowest FZI, lowest permeability)
  - **Cluster 8** = best quality rock (highest FZI, highest permeability)

### 5. SDR Model Calibration
The generalized SDR model is linearized via log₁₀ transformation:

```
log₁₀(k) = C_m + α · log₁₀(Φ) + β · log₁₀(T₂gm)
```

- Fitted using **Ordinary Least Squares (OLS)** via `statsmodels`
- Two variants: **(1)** global model on all data, **(2)** cluster-specific model per HFU
- **80/20 train-test split** (random seed = 2)
- **Bootstrap resampling (R = 10,000)** for coefficient uncertainty quantification

### 6. Evaluation
Performance is measured using RMSE on the log₁₀ scale:

```
RMSE(log₁₀ k) = √[ (1/n) Σ (log₁₀(k_pred + ε) − log₁₀(k_true + ε))² ]
```

where `ε = 1e-12` prevents undefined values. This penalizes relative errors uniformly across the permeability range. R² is also reported.

---

## Key Results

| Group | RMSE Train | RMSE Test | R² Train | R² Test |
|-------|-----------|----------|---------|--------|
| No Clustering | 1.062 | 1.212 | −0.121 | −0.163 |
| Cluster 1 | 0.451 | 0.374 | 0.151 | 0.077 |
| Cluster 2 | 0.432 | 0.491 | 0.084 | 0.231 |
| Cluster 3 | 0.665 | 0.761 | 0.007 | −0.088 |
| Cluster 4 | 0.679 | 0.635 | 0.001 | −0.137 |
| Cluster 5 | 0.851 | 0.899 | −0.051 | −0.156 |
| Cluster 6 | 0.715 | 0.918 | 0.122 | −0.186 |
| Cluster 7 | 0.733 | 1.367 | 0.013 | −0.260 |
| Cluster 8 | 0.321 | 0.839 | −2.280 | 0.167 |

The global model's negative R² (−0.163) confirms it performs **worse than predicting the mean**. Cluster-specific models reduce test RMSE(log₁₀) by up to **69%** for the best-performing cluster.

---

## Requirements

**Core dependencies:**

```
numpy
pandas
matplotlib
scikit-learn
statsmodels
jupyter
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels jupyter
```

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nihal-ml/nmr-permeability.git
   cd nmr-permeability
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib scikit-learn statsmodels jupyter
   ```

3. **Launch the notebook:**
   ```bash
   jupyter notebook Research.ipynb
   ```

4. **Run all cells top to bottom.** All data files are loaded using relative paths — keep all CSVs in the same directory as the notebook.

---

## References

- Amaefule, J.O. et al. (1993). Enhanced Reservoir Description Using Core and Log Data to Identify Hydraulic (Flow) Units. *SPE-26436-MS.*
- Kenyon, W.E. et al. (1988). A Three-Part Study of NMR Longitudinal Relaxation Properties of Water-Saturated Sandstones. *SPE Formation Evaluation*, 3(3), 622–636.
- Coates, G.R., Xiao, L., and Prammer, M.G. (1999). *NMR Logging: Principles and Applications.* Halliburton Energy Services.
- Parsekian, A.D. et al. (2015). Uncertainty in the Determination of Soil Hydraulic Parameters. *Water Resources Research*, 51(5).
- Knight, R. et al. (2016). NMR Logging to Estimate Hydraulic Conductivity in Unconsolidated Aquifers. *Geophysical Research Letters*, 43(9), 4367–4374.

---

## License

This project is submitted as academic coursework for DASC 5391 at Kansas State University.
