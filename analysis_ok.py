"""
O K edge EELS analysis: A/B peak ratio and oxygen vacancy mapping.

Workflow
--------
1. Preprocess: energy crop [500, 560 eV], stride-2 spatial binning
2. PCA denoising (fixed 4 components)
3. Savitzky-Golay smoothing
4. Power-law background subtraction
5. Automatic A and B peak detection via local maximum / zero-crossing
6. Pixel-wise I(A)/I(B) ratio and Vo calibration
7. K-means background mask (3 clusters on integrated intensity)
8. Save: A/B map, Vo map, denoised signal
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from sklearn.cluster import KMeans
from scipy.ndimage import maximum_filter

from preprocessing import (
    preprocess_signal,
    denoise_signal,
    smooth_signal,
    subtract_background_signal,
)
from config import (
    OK_ENERGY_MIN, OK_ENERGY_MAX, OK_STRIDE_BINNING,
    OK_N_PCA, OK_KMEANS_CLUSTERS, OK_KMEANS_MAX_ITER,
    VO_INTERCEPT, VO_SLOPE,
)


def perform_ok_analysis(data_sig, filepath, save=True):
    """Run O K edge analysis on a loaded EELS signal.

    Parameters
    ----------
    data_sig : HyperSpy EELSSpectrum
    filepath : str
        Path to the source file; used to derive the output directory.
    save : bool
        Write output files when True.
    """
    dst_dir = os.path.dirname(filepath)

    print("Preprocessing O K spectrum (energy crop + stride binning)...")
    EELS_sig = preprocess_signal(
        data_sig,
        energy_min=OK_ENERGY_MIN,
        energy_max=OK_ENERGY_MAX,
        stride_binning=OK_STRIDE_BINNING,
    )

    print(f"Denoising with {OK_N_PCA} PCA components (fixed)...")
    EELS_sig = denoise_signal(EELS_sig, method="PCA", n_components=OK_N_PCA)

    print("Smoothing signal (Savitzky-Golay)...")
    EELS_sig = smooth_signal(EELS_sig)

    print("Subtracting background (power-law fit)...")
    EELS_sig_sub = subtract_background_signal(EELS_sig)
    EELS_sig_sub.plot()
    plt.show()

    A_peak_pos, B_peak_pos = _detect_peaks(EELS_sig_sub)
    if A_peak_pos is None:
        return

    irox_mask = _make_particle_mask(EELS_sig_sub)
    AB_map, vo_map = _compute_ratio_maps(EELS_sig_sub, A_peak_pos, B_peak_pos)

    AB_map_final = hs.signals.Signal2D(AB_map * irox_mask)
    vo_map_final = hs.signals.Signal2D(vo_map * irox_mask)

    _plot_ok_results(EELS_sig_sub, AB_map_final, vo_map_final)

    if save:
        _save_ok_results(dst_dir, AB_map_final, vo_map_final, EELS_sig_sub)
        print(f"Results saved to: {dst_dir}/OK_results/")
    else:
        print("O K analysis complete. Results not saved (--no-save).")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_peaks(sig_sub):
    """Locate B peak (dominant maximum) and A peak (30 channels before B).

    Returns (A_peak_pos, B_peak_pos) as integer channel indices,
    or (None, None) on failure.
    """
    sum_data = sig_sub.sum().data
    try:
        peak_indices = np.where(maximum_filter(sum_data, 101) == sum_data)[0]
        B_peak_pos = int(peak_indices[np.argmin(np.abs(peak_indices - 340))])
        A_peak_pos = B_peak_pos - 30
        return A_peak_pos, B_peak_pos
    except (ValueError, IndexError):
        print("ERROR: Failed to locate O-K edge peaks. Check the data energy range.")
        return None, None


def _make_particle_mask(sig_sub):
    """Return a binary (0/1) mask isolating IrOx particle pixels
    via K-means on integrated intensity."""
    sig_sum = sig_sub.data.sum(axis=2)
    km = KMeans(n_clusters=OK_KMEANS_CLUSTERS,
                n_init="auto", max_iter=OK_KMEANS_MAX_ITER)
    labels = km.fit_predict(sig_sum.reshape(-1, 1))

    mean_intensity = [sig_sum.flatten()[labels == i].mean()
                      for i in range(OK_KMEANS_CLUSTERS)]
    bg_label = int(np.argmin(mean_intensity))
    return (labels != bg_label).reshape(sig_sum.shape).astype(float)


def _compute_ratio_maps(sig_sub, A_peak_pos, B_peak_pos):
    """Compute pixel-wise I(A)/I(B) ratio and convert to vacancy concentration.

    A peak: minimum of |dI/dE| in [A_peak_pos-5, A_peak_pos+5]
    B peak: maximum of I in [B_peak_pos-10, B_peak_pos+10]

    Returns
    -------
    AB_map : ndarray, shape (H, W)
    vo_map : ndarray, shape (H, W)
    """
    diff = np.gradient(sig_sub.data, axis=2)
    a_range = slice(A_peak_pos - 5, A_peak_pos + 6)
    b_range = slice(B_peak_pos - 10, B_peak_pos + 11)

    A_pos_map = np.argmin(np.abs(diff[:, :, a_range]), axis=-1) + a_range.start
    B_pos_map = np.argmax(sig_sub.data[:, :, b_range], axis=-1) + b_range.start

    h, w, _ = sig_sub.data.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)
    I_A = sig_sub.data[rows, cols, A_pos_map]
    I_B = sig_sub.data[rows, cols, B_pos_map]

    AB_map = np.divide(I_A, I_B, out=np.zeros_like(I_A), where=(I_B != 0))
    vo_map = np.clip((AB_map - VO_INTERCEPT) / VO_SLOPE, 0, None)
    return AB_map, vo_map


def _plot_ok_results(sig_sub, AB_map_final, vo_map_final):
    sig_sum = sig_sub.data.sum(axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sig_sum, cmap="gray")
    axes[0].set_title("Background-subtracted Intensity")
    axes[0].axis("off")

    valid = AB_map_final.data[AB_map_final.data > 0]
    vmin, vmax = (np.percentile(valid, [5, 95]) if valid.size > 0 else (0, 1))
    im1 = axes[1].imshow(AB_map_final.data, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("O K A/B Ratio Map")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(vo_map_final.data, cmap="plasma")
    axes[2].set_title("O K Vo Map")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def _save_ok_results(dst_dir, AB_map_final, vo_map_final, EELS_sig_sub):
    result_folder = os.path.join(dst_dir, "OK_results")
    os.makedirs(result_folder, exist_ok=True)

    AB_map_final.save(
        os.path.join(result_folder, "ab_ratio_map.tif"), overwrite=True
    )
    vo_map_final.save(
        os.path.join(result_folder, "vo_map.tif"), overwrite=True
    )
    EELS_sig_sub.T.save(
        os.path.join(result_folder, "denoised_signal.tif"), overwrite=True
    )
