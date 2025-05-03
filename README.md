# Order Flow Imbalance (OFI) Feature Engineering

This repository contains Python implementations of Order Flow Imbalance (OFI) features based on the task specification from a quantitative research project. The OFI measures are critical for modeling short-term price impact in equity markets.

Features Implemented

1. Best-Level OFI
- Calculates net order flow at the **best bid and ask prices (Level 0)**.
- Rolling aggregation over a time window `(t - h, t]`.
- One OFI value is computed for each timestamp in the dataset.

2. Multi-Level OFI
- Extends OFI to **multiple LOB depth levels (Levels 0–9)**.
- Captures additional liquidity information beyond the top-of-book.
- Returns a feature vector of OFI values per level for each timestamp.

3. Integrated OFI
- Applies **Principal Component Analysis (PCA)** on normalized multi-level OFIs.
- The first principal component is used to construct a single integrated OFI signal:
  \[
  \text{Integrated OFI}_{t} = \frac{\mathbf{w}_1^T \cdot \mathbf{ofi}_{t}}{\|\mathbf{w}_1\|_1}
  \]

Files

- `ofi_pipeline.py`: Contains all code for loading the data, computing best-level, multi-level, and integrated OFI features.
- `first_25000_rows.csv`: Input dataset (not included in repo due to size/privacy — upload manually).
- `ofi_features_output.csv`: Output CSV file with all computed OFI features (generated after running the script).

Requirements

- Python 3.7+
- `pandas`
- `numpy`
- `scikit-learn`

Install dependencies:

```bash
pip install pandas numpy scikit-learn
