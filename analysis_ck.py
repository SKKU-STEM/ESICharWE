"""
C K edge EELS analysis: NMF decomposition + K-means clustering.

Workflow
--------
1. Preprocess: energy crop [270, 330 eV], stride-2 spatial binning
2. NMF: user-defined number of components
3. K-means: cluster NMF loading vectors
4. Mask: suppress background pixels below mean intensity
5. Save: loading maps, loading vectors, cluster masks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

from preprocessing import preprocess_signal
from config import (
    CK_ENERGY_MIN, CK_ENERGY_MAX, CK_STRIDE_BINNING,
    CK_NMF_MAX_ITER, CK_KMEANS_MAX_ITER,
)


def perform_ck_analysis(data_sig, filepath, n_components, save=True):
    """Run C K edge analysis on a loaded EELS signal.

    Parameters
    ----------
    data_sig : HyperSpy EELSSpectrum
    filepath : str
        Path to the source file; used to derive the output directory.
    n_components : int
        Number of NMF / K-means components (2-10).
    save : bool
        Write output files when True.
    """
    dst_dir = os.path.dirname(filepath)

    print("Preprocessing C K spectrum (energy crop + stride binning)...")
    EELS_sig = preprocess_signal(
        data_sig.copy(),
        energy_min=CK_ENERGY_MIN,
        energy_max=CK_ENERGY_MAX,
        normalize=False,
        stride_binning=CK_STRIDE_BINNING,
    )
    EELS_sig.plot()
    plt.show()

    print(f"Performing NMF with {n_components} components...")
    data_arr = EELS_sig.data.copy()
    h, w, c = data_arr.shape
    data_flat = data_arr.reshape(-1, c)
    data_flat[data_flat < 0] = 0

    nmf_model = NMF(n_components=n_components,
                    init="nndsvd", max_iter=CK_NMF_MAX_ITER)
    W = nmf_model.fit_transform(data_flat)
    H = nmf_model.components_

    W_sig, H_sig = _build_hyperspy_signals(W, H, EELS_sig, h, w)
    _plot_nmf_results(W_sig, H_sig, n_components)

    print("Performing K-means clustering on NMF loadings...")
    label_map_data = _run_kmeans(W, h, w, EELS_sig, n_components)
    label_map = hs.signals.Signal2D(label_map_data)
    _plot_cluster_map(label_map, n_components)

    if save:
        _save_ck_results(dst_dir, n_components, H_sig, W_sig, label_map)
        print(f"Results saved to: {dst_dir}/CK_results_NMF_{n_components}/")
    else:
        print("C K analysis complete. Results not saved (--no-save).")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_hyperspy_signals(W, H, ref_sig, h, w):
    """Wrap NMF W and H matrices into HyperSpy signal objects."""
    n = H.shape[0]
    W_sig = hs.signals.Signal2D(W.T.reshape(n, h, w))

    H_sig = hs.signals.Signal1D(H.reshape(n, H.shape[-1]))
    ax = ref_sig.axes_manager[-1]
    for attr in ("scale", "units", "offset"):
        setattr(H_sig.axes_manager[-1], attr, getattr(ax, attr))
    H_sig.change_dtype("float32")

    return W_sig, H_sig


def _plot_nmf_results(W_sig, H_sig, n_components):
    """Display NMF loading maps and corresponding spectra."""
    fig, axes = plt.subplots(2, n_components, figsize=(4 * n_components, 7))
    fig.suptitle(
        "NMF Components and Spectra\n"
        "(close this window to proceed to K-means clustering)",
        fontsize=14,
    )
    for i in range(n_components):
        ax_w = axes[0, i]
        im = ax_w.imshow(W_sig.data[i], cmap="viridis")
        ax_w.set_title(f"Component {i + 1}")
        ax_w.axis("off")
        fig.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)

        ax_h = axes[1, i]
        ax_h.plot(H_sig.axes_manager[-1].axis, H_sig.data[i],
                  color="red", linewidth=1.5)
        ax_h.set_xlabel(f"Energy loss ({H_sig.axes_manager[-1].units})")
        ax_h.set_ylabel("Intensity")
        ax_h.grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def _run_kmeans(W, h, w, EELS_sig, n_components):
    """Cluster NMF loadings and mask background pixels."""
    km = KMeans(n_clusters=n_components, n_init="auto", max_iter=CK_KMEANS_MAX_ITER)
    labels = km.fit_predict(W).reshape(h, w)

    intensity_map = EELS_sig.data.sum(axis=2)
    particle_mask = (intensity_map >= intensity_map.mean()).astype(int)
    return labels * particle_mask


def _plot_cluster_map(label_map, n_components):
    cmap = plt.get_cmap("viridis", n_components)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(label_map.data + 1, cmap=cmap)
    ax.set_title("Cluster Label Map")
    ax.axis("off")
    cbar = fig.colorbar(im, ticks=np.arange(n_components) + 1)
    cbar.set_label("Cluster Index")
    plt.tight_layout()
    plt.show()


def _save_ck_results(dst_dir, n_components, H_sig, W_sig, label_map):
    result_folder = os.path.join(dst_dir, f"CK_results_NMF_{n_components}")
    os.makedirs(result_folder, exist_ok=True)

    header = ",".join([f"loading_vector_{i + 1}" for i in range(n_components)])
    np.savetxt(
        os.path.join(result_folder, "1_NMF_loading_vector.csv"),
        H_sig.data.T,
        delimiter=",", header=header, comments="",
    )

    W_sig.save(os.path.join(result_folder, "1_NMF_loading_map.tif"), overwrite=True)
    for i in range(n_components):
        hs.signals.Signal2D(W_sig.data[i]).save(
            os.path.join(result_folder, f"1_NMF_loading_map_{i + 1}.tif"),
            overwrite=True,
        )

    label_map.change_dtype("uint8")
    label_map.save(
        os.path.join(result_folder, "2_KMeans_cluster_map.tif"), overwrite=True
    )
    for i in range(n_components):
        mask = (label_map.data == i).astype(np.uint8)
        hs.signals.Signal2D(mask).save(
            os.path.join(result_folder, f"2_KMeans_cluster_{i + 1}_mask.tif"),
            overwrite=True,
        )
