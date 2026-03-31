"""
Signal preprocessing utilities for STEM-EELS data.

All functions accept and return HyperSpy Signal1D/EELSSpectrum objects
(or compatible objects with a .data attribute of shape (H, W, E)).
"""

import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

from config import (
    OK_SAVGOL_WINDOW,
    OK_SAVGOL_POLYORDER,
    OK_BG_FIT_RANGE,
)


def flatten_signal(sig):
    """Reshape (H, W, E) data to (H*W, E) for matrix decomposition.

    Returns
    -------
    flat : ndarray, shape (H*W, E)
    shape : tuple (H, W)
    """
    h, w, e = sig.data.shape
    flat = sig.data.reshape(-1, e)
    return flat, (h, w)


def preprocess_signal(sig, energy_min=None, energy_max=None,
                      normalize=False, stride_binning=None):
    """Apply energy-range cropping, stride-based spatial binning,
    and optional per-spectrum normalization.

    Parameters
    ----------
    sig : HyperSpy signal
    energy_min, energy_max : float, optional
        Crop the energy axis to [energy_min, energy_max].
    normalize : bool
        Divide each spectrum by its maximum value.
    stride_binning : int, optional
        Sum overlapping (stride_binning x stride_binning) spatial patches.
        Output spatial size becomes (H - k + 1, W - k + 1).

    Returns
    -------
    sig : HyperSpy signal (modified copy)
    """
    sig.data = np.abs(sig.data)

    if energy_min is not None and energy_max is not None:
        sig = sig.isig[float(energy_min):float(energy_max)]

    if stride_binning is not None:
        data = sig.data
        H, W, E = data.shape
        k = stride_binning
        new_H = H - k + 1
        new_W = W - k + 1
        binned_data = np.zeros((new_H, new_W, E), dtype=data.dtype)
        for i in range(new_H):
            for j in range(new_W):
                binned_data[i, j] = data[i:i + k, j:j + k, :].sum(axis=(0, 1))
        sig = sig._deepcopy_with_new_data(binned_data)
        sig.axes_manager[0].size = new_W
        sig.axes_manager[1].size = new_H

    if normalize:
        sig.data /= np.max(sig.data, axis=-1, keepdims=True)

    return sig


def denoise_signal(sig, method="PCA", n_components=3):
    """Reconstruct the signal from the leading PCA components
    to suppress high-frequency noise.

    Parameters
    ----------
    sig : HyperSpy signal, shape (H, W, E)
    method : str
        Decomposition method. Currently only 'PCA' is supported.
    n_components : int
        Number of components to retain.

    Returns
    -------
    sig_denoised : HyperSpy signal
    """
    flat, shape = flatten_signal(sig)
    if method == "PCA":
        model = PCA(n_components=n_components)
    else:
        raise ValueError(f"Unsupported denoising method: '{method}'. Use 'PCA'.")

    comp = model.fit_transform(flat)
    recon = model.inverse_transform(comp)
    return sig._deepcopy_with_new_data(recon.reshape(shape + (-1,)))


def smooth_signal(sig, window_length=OK_SAVGOL_WINDOW,
                  polyorder=OK_SAVGOL_POLYORDER):
    """Apply a Savitzky-Golay filter along the energy axis.

    Parameters
    ----------
    sig : HyperSpy signal, data.ndim must be 3 (H, W, E)
    window_length : int
        Length of the filter window (must be odd).
    polyorder : int
        Polynomial order used to fit within each window.

    Returns
    -------
    HyperSpy signal with smoothed data
    """
    if sig.data.ndim != 3:
        raise ValueError("Signal data must be 3-dimensional (H, W, E).")
    smoothed = savgol_filter(sig.data, window_length, polyorder, axis=-1)
    return sig._deepcopy_with_new_data(smoothed)


def subtract_background_signal(sig, fit_range=OK_BG_FIT_RANGE):
    """Remove the pre-edge power-law background pixel by pixel.

    A power-law model  A * E^(-r)  is fitted to the energy range
    [fit_range[0], fit_range[1]] and subtracted from the full spectrum.
    Pixels where the fit fails fall back to the uncorrected spectrum.

    Parameters
    ----------
    sig : HyperSpy signal, shape (H, W, E)
    fit_range : tuple of float
        (E_min, E_max) in eV used for background fitting.

    Returns
    -------
    sig_corrected : HyperSpy signal
    """
    def _power_law(x, A, r):
        return A * np.power(x, -r)

    energy_axis = sig.axes_manager[-1].axis
    h, w, _ = sig.data.shape
    corrected = np.empty_like(sig.data)
    mask = (energy_axis >= fit_range[0]) & (energy_axis <= fit_range[1])
    x_fit = energy_axis[mask]

    for i in tqdm(range(h), desc="Background subtraction"):
        for j in range(w):
            y = sig.data[i, j, :]
            y_fit = np.maximum(y[mask], 1e-6)
            try:
                popt, _ = curve_fit(_power_law, x_fit, y_fit,
                                    p0=(1.0, 2.0), maxfev=5000)
                corrected[i, j, :] = y - _power_law(energy_axis, *popt)
            except RuntimeError:
                corrected[i, j, :] = y

    return sig._deepcopy_with_new_data(corrected)
