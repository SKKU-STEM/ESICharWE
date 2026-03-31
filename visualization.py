"""
Multi-map overlay visualization.

Combines an ADF image, C K cluster masks (ionomer / IrOx),
and the O K oxygen-vacancy map into a single composite figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import resize
from skimage.color import rgb2gray


def combine_map(adf, ck_ionomer, ck_irox, vacancy_map):
    """Overlay ADF, C K phase masks, and O K vacancy map.

    Parameters
    ----------
    adf : ndarray
        ADF image (grayscale or RGB structured array with fields R/G/B).
    ck_ionomer : ndarray, shape (H, W)
        Binary (0/1) mask for the ionomer phase.
    ck_irox : ndarray, shape (H, W)
        Binary (0/1) mask for the IrOx particle phase.
    vacancy_map : ndarray, shape (H, W)
        Oxygen vacancy concentration map; zero/negative values are masked.
    """
    ionomer_pix = int(np.count_nonzero(ck_ionomer))
    irox_pix = int(np.count_nonzero(ck_irox))
    denom = ionomer_pix + irox_pix
    coverage = ionomer_pix / denom if denom > 0 else 0.0

    # Colormaps
    ck_ionomer_cmap = LinearSegmentedColormap.from_list(
        "black_yellow", [(0, 0, 0), (1, 1, 0)]
    )
    ck_irox_cmap = LinearSegmentedColormap.from_list(
        "black_purple", [(0, 0, 0), (1, 0, 1)]
    )
    vac_cmap = LinearSegmentedColormap.from_list(
        "blue_green_yellow",
        [
            (0.0, (0.0, 0.1, 0.8)),
            (0.3, (0.0, 0.6, 0.5)),
            (0.5, (0.0, 0.8, 0.4)),
            (1.0, (1.0, 1.0, 0.0)),
        ],
    )
    masked_vac = np.ma.masked_where(vacancy_map <= 0, vacancy_map)
    vac_cmap.set_bad(color="black")

    adf_gray = _to_grayscale(adf)
    new_h, new_w = ck_ionomer.shape
    adf_resized = resize(adf_gray, (new_h, new_w))

    # Layout: ADF+masks | ADF+vacancy | colorbar | ADF+combined
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 0.05, 1], wspace=0.1)

    ax1 = plt.subplot(gs[0])
    ax1.imshow(adf_resized, cmap="gray")
    ax1.imshow(ck_ionomer, cmap=ck_ionomer_cmap, alpha=0.2, vmin=0, vmax=1)
    ax1.imshow(ck_irox, cmap=ck_irox_cmap, alpha=0.2, vmin=0, vmax=1)
    ax1.set_title(
        f"ADF + Ionomer + Particle\nIonomer coverage: {coverage:.2%}", fontsize=14
    )
    ax1.axis("off")

    ax2 = plt.subplot(gs[1])
    ax2.imshow(adf_resized, cmap="gray")
    vac_im = ax2.imshow(masked_vac, cmap=vac_cmap, alpha=0.3, vmin=0, vmax=10)
    ax2.set_title("ADF + Oxygen Vacancy Map (%)", fontsize=14)
    ax2.axis("off")

    cax = plt.subplot(gs[2])
    plt.colorbar(vac_im, cax=cax)

    ax3 = plt.subplot(gs[3])
    ax3.imshow(adf_resized, cmap="gray")
    ax3.imshow(masked_vac, cmap=vac_cmap, vmin=0, vmax=10, alpha=0.5)
    ax3.imshow(ck_ionomer, cmap=ck_ionomer_cmap, alpha=0.1, vmin=0, vmax=1)
    ax3.imshow(ck_irox, cmap=ck_irox_cmap, alpha=0.3, vmin=0, vmax=1)
    ax3.set_title("Combined Map", fontsize=14)
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def _to_grayscale(adf):
    """Convert an ADF image to a 2-D float grayscale array.

    Handles two input formats:
    - NumPy structured array with fields 'R', 'G', 'B'
    - Plain 2-D or 3-D ndarray
    """
    if (
        hasattr(adf, "dtype")
        and adf.dtype.names is not None
        and all(c in adf.dtype.names for c in ("R", "G", "B"))
    ):
        rgb = np.stack([adf["R"], adf["G"], adf["B"]], axis=-1)
        return rgb2gray(rgb)
    return adf
