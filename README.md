# ESICharWE

## Machine-learning-driven integrated probing of oxygen-vacancy distribution and ionomer morphology in iridium oxide catalyst–ionomer nanocomposite electrode for water electrolyzer

Yerin Jeon†, Sang-Hyeok Yang†, Hyeon-Ah Ju†, Kwanhong Park†, Wooseon Choi, Daehee Yang, Hakjoo Lee, Dami Lim, Shin Jang, Jaekwang Lee*, Jae-Hyeok Kim* and Young-Min Kim\*

Published in \_\_

---

## File Structure

```
eels_mapping/
├── main.py           # Entry point (interactive CLI)
├── config.py         # All analysis parameters
├── preprocessing.py  # Energy crop, binning, PCA denoising, smoothing, BG subtraction
├── analysis_ck.py    # C K edge: NMF + K-means
├── analysis_ok.py    # O K edge: A/B ratio + Vo calibration
├── visualization.py  # Multi-layer ADF/phase/vacancy overlay
└── requirements.txt
```

---

## Requirements

- Python 3.9+

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

Program prompts:

```
==================================================
  STEM-EELS Mapping Program
==================================================
  1. C K edge analysis (NMF + K-means)
  2. O K edge analysis (A/B ratio + Vo map)
  3. Combined map overlay
  q. Quit
==================================================
Select menu: _
```

### C K edge analysis (menu: 1)

| Prompt                   | Default | Description                        |
| ------------------------ | ------- | ---------------------------------- |
| EELS file path           | —       | `.dm3` / `.dm4` file               |
| Energy-axis offset (eV)  | 220.0   | Asked only when stored offset <= 0 |
| Number of NMF components | 3       | Integer 2–10                       |
| Save results to disk     | Y       | y/n                                |

Output folder: `<source_dir>/CK_results_NMF_<N>/`

| File                            | Contents                         |
| ------------------------------- | -------------------------------- |
| `1_NMF_loading_vector.csv`      | NMF H matrix (component spectra) |
| `1_NMF_loading_map.tif`         | All loading maps stacked         |
| `1_NMF_loading_map_<i>.tif`     | Individual loading maps          |
| `2_KMeans_cluster_map.tif`      | Labelled cluster image           |
| `2_KMeans_cluster_<i>_mask.tif` | Binary mask per cluster          |

### O K edge analysis (menu: 2)

| Prompt                  | Default | Description                        |
| ----------------------- | ------- | ---------------------------------- |
| EELS file path          | —       | `.dm3` / `.dm4` file               |
| Energy-axis offset (eV) | 220.0   | Asked only when stored offset <= 0 |
| Save results to disk    | Y       | y/n                                |

Output folder: `<source_dir>/OK_results/`

| File                  | Contents                                     |
| --------------------- | -------------------------------------------- |
| `ab_ratio_map.tif`    | Pixel-wise I(A)/I(B) ratio                   |
| `vo_map.tif`          | Oxygen vacancy concentration (%)             |
| `denoised_signal.tif` | PCA + Savitzky-Golay filtered spectrum image |

### Combined map overlay (menu: 3)

| Prompt                | Description                     |
| --------------------- | ------------------------------- |
| ADF image path        | Grayscale or RGB `.tif`         |
| C K ionomer mask path | Binary `.tif` from C K analysis |
| C K IrOx mask path    | Binary `.tif` from C K analysis |
| O K vacancy map path  | Float `.tif` from O K analysis  |

---

## Analysis Parameters

All parameters are in `config.py`.

| Parameter                   | Default       | Description                      |
| --------------------------- | ------------- | -------------------------------- |
| `CK_ENERGY_MIN/MAX`         | 270 / 330 eV  | C K energy window                |
| `OK_ENERGY_MIN/MAX`         | 500 / 560 eV  | O K energy window                |
| `OK_N_PCA`                  | 4             | PCA components for O K denoising |
| `OK_BG_FIT_RANGE`           | (500, 528) eV | Power-law background fit range   |
| `VO_INTERCEPT` / `VO_SLOPE` | 0.492 / 0.017 | Vo = (A/B − 0.492) / 0.017       |

---

## Data Format

- Input: DigitalMicrograph `.dm3` / `.dm4` EELS spectrum images
- Spike noise should be removed in DigitalMicrograph before use
- ADF images for overlay should be exported as `.tif`

---

## Contact

rb4738@g.skku.edu
