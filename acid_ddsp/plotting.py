import io
import logging
import os
from typing import Optional, List

import PIL
import librosa
import librosa.display
import numpy as np
import torch as tr
from matplotlib import pyplot as plt
from matplotlib.axes import Subplot
from matplotlib.figure import Figure
from torch import Tensor as T
from torchvision.transforms import ToTensor
from tqdm import tqdm

from paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def fig2img(fig: Figure, format: str = "png", dpi: int = 120) -> T:
    """Convert a matplotlib figure to tensor."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close("all")
    return image


def plot_scalogram(
    ax: Subplot,
    scalogram: T,
    sr: float,
    y_coords: List[float],
    title: Optional[str] = None,
    hop_len: int = 1,
    cmap: str = "magma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    x_label: str = "time (s)",
    y_label: str = "freq (Hz)",
    fontsize: int = 12,
) -> None:
    """
    Plots a scalogram of the provided data.

    The scalogram is a visual representation of the wavelet transform of a signal over time.
    This function uses matplotlib and librosa to create the plot.

    Parameters:
        ax (Subplot): The axis on which to plot the scalogram.
        scalogram (T): The scalogram data to be plotted.
        sr (float): The sample rate of the audio signal.
        y_coords (List[float]): The y-coordinates for the scalogram plot.
        title (str, optional): The title of the plot. Defaults to "scalogram".
        hop_len (int, optional): The hop length for the time axis (or T). Defaults to 1.
        cmap (str, optional): The colormap to use for the plot. Defaults to "magma".
        vmax (Optional[float], optional): The maximum value for the colorbar. If None,
            the colorbar scales with the data. Defaults to None.
        x_label (str, optional): The label for the x-axis. Defaults to "Time (seconds)".
        y_label (str, optional): The label for the y-axis. Defaults to "Frequency (Hz)".
        fontsize (int, optional): The fontsize for the labels. Defaults to 16.
    """
    assert scalogram.ndim == 2
    assert scalogram.size(0) == len(y_coords)
    if vmin is not None and vmax is not None:
        # This is supposed to prevent an IndexError when vmin == vmax == 0.0
        if vmin == vmax:
            vmin = None
            vmax = None
        assert vmin < vmax
    x_coords = librosa.times_like(scalogram.size(1), sr=sr, hop_length=hop_len)

    try:
        librosa.display.specshow(
            ax=ax,
            data=scalogram.numpy(),
            sr=sr,
            x_axis="time",
            x_coords=x_coords,
            y_axis="cqt_hz",
            y_coords=np.array(y_coords),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    except IndexError as e:
        # TODO(cm): fix this
        log.warning(f"IndexError: {e}")

    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    if len(y_coords) < 12:
        ax.set_yticks(y_coords)
    ax.minorticks_off()
    if title:
        ax.set_title(title, fontsize=fontsize)


def plot_waveforms_stacked(
    waveforms: List[T],
    sr: float,
    title: Optional[str] = None,
    waveform_labels: Optional[List[str]] = None,
    show: bool = False,
) -> Figure:
    assert waveforms
    if waveform_labels is None:
        waveform_labels = [None] * len(waveforms)
    assert len(waveform_labels) == len(waveforms)

    fig, axs = plt.subplots(
        nrows=len(waveforms),
        sharex="all",
        sharey="all",
        figsize=(7, 2 * len(waveforms)),
        squeeze=False,
    )
    axs = axs.squeeze(1)

    for idx, (ax, w, label) in enumerate(zip(axs, waveforms, waveform_labels)):
        assert 0 < w.ndim <= 2
        if w.ndim == 2:
            assert w.size(0) == 1
            w = w.squeeze(0)
        w = w.detach().float().cpu().numpy()
        if idx == len(waveforms) - 1:
            axis = "time"
        else:
            axis = None
        librosa.display.waveshow(w, axis=axis, sr=sr, label=label, ax=ax)
        ax.set_title(label)
        ax.grid(color="lightgray", axis="x")
        # ax.set_xticks([])
        # ax.set_yticks([])

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    # fig.savefig(os.path.join(OUT_DIR, f"3.svg"))

    if show:
        fig.show()
    return fig


def plot_xy_points_and_grads(
    ax: Subplot,
    x: T,
    y: T,
    x_hat: T,
    y_hat: T,
    x_grad: Optional[T] = None,
    y_grad: Optional[T] = None,
    title: str = "",
    x_label: str = "θ slope",
    y_label: str = "θ density",
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    fontsize: int = 12,
):
    for idx in range(len(x)):
        ax.plot(
            [x_hat[idx], x[idx]],
            [y_hat[idx], y[idx]],
            color="lightgrey",
            linestyle="dashed",
            linewidth=1,
            zorder=0,
        )
    if x_grad is not None and y_grad is not None:
        ax.quiver(
            x_hat.numpy(),
            y_hat.numpy(),
            -x_grad.numpy(),
            -y_grad.numpy(),
            color="red",
            angles="xy",
            scale=5.0,
            scale_units="width",
            zorder=1,
        )
    ax.scatter(x_hat, y_hat, color="black", marker="o", facecolors="none", zorder=2)
    ax.scatter(x, y, color="black", marker="x", zorder=2)
    ax.set_xlim(xmin=x_min, xmax=x_max)
    ax.set_ylim(ymin=y_min, ymax=y_max)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)


def piecewise_fitting_noncontinuous(
    y: np.ndarray,
    degree: int = 3,
    n_knots: int = 0,
    knot_locations: Optional[List[int]] = None,
) -> np.ndarray:
    assert y.ndim == 1
    n_samples = y.shape[0]
    if knot_locations is None:
        step_size = n_samples // (n_knots + 1)
        knot_locations = [step_size * (idx + 1) for idx in range(n_knots)]
    assert len(knot_locations) == n_knots
    assert all(0 < k < n_samples for k in knot_locations)

    segments = []
    knot_locations = knot_locations + [n_samples]
    x = np.linspace(0, 1, n_samples)
    start_idx = 0
    for end_idx in knot_locations:
        curr_x = x[start_idx:end_idx]
        curr_y = y[start_idx:end_idx]
        poly = np.polynomial.Polynomial.fit(x=curr_x, y=curr_y, deg=degree)
        segment = poly(curr_x)
        segments.append(segment)
        start_idx = end_idx

    y_fitted = np.concatenate(segments)
    return y_fitted


# def plot_waterfall(
#     ax: Subplot,
#     sr: int,
#     H: T,
#     n_fft: int,
#     title: str = "H",
#     min_t: int = 0,
#     max_t: Optional[int] = None,
#     use_log: bool = False,
#     freq_offset: int = 20,
#     spec_offset: int = 10,
# ) -> None:
#     assert H.ndim == 2
#     if max_t is None:
#         max_t = H.size(1)
#     min_t = min(min_t, H.size(1))
#     max_t = min(max_t, H.size(1))
#     H = H.abs()
#     if use_log:
#         H = H.log1p()
#     freqs = np.arange(H.size(0)) / n_fft * sr
#     H_plot = H[:, min_t:max_t] - np.arange(max_t - min_t) * spec_offset
#     H_plot_min = H_plot.min()
#     for i in range(max_t - min_t):
#         zorder = i
#         ax.plot(freqs - i * freq_offset, H_plot[:, i], "k", zorder=zorder)
#         ax.fill_between(
#             freqs - i * freq_offset,
#             H_plot_min,
#             H[:, i],
#             facecolor="white",
#             zorder=zorder,
#         )
#     ax.set_title(title, fontsize=18)
#     ax.axis("off")


def plot_waterfall_og(
    ax: Subplot,
    sr: int,
    H: T,
    n_fft: int,
    title: str = "H",
    min_t: int = 0,
    max_t: Optional[int] = None,
    use_log: bool = False,
    freq_offset: int = 20,
    spec_offset: int = 10,
) -> None:
    assert H.ndim == 2
    H = H.swapaxes(0, 1)
    if max_t is None:
        max_t = H.size(1)
    min_t = min(min_t, H.size(1))
    max_t = min(max_t, H.size(1))
    H = H.abs()
    if use_log:
        H += 1e-7
        H = H.log()
        H *= 10

    freqs = np.arange(H.size(0)) / n_fft * sr
    H_plot = H[:, min_t:max_t] - np.arange(max_t - min_t) * spec_offset
    H_plot_min = H_plot.min()

    for idx in range(max_t - min_t):
        zorder = idx
        ax.plot(freqs - idx * freq_offset, H_plot[:, idx], "k", zorder=zorder)
        ax.fill_between(
            freqs - idx * freq_offset,
            H_plot_min,
            H_plot[:, idx],
            facecolor="white",
            zorder=zorder,
        )
    ax.set_title(title, fontsize=18)
    ax.axis("off")


def plot_waterfalls(H_paths: List[str], idx: int, use_log: bool = False) -> Figure:
    n_plots = len(H_paths)
    fig, axs = plt.subplots(
        1, n_plots, figsize=(4 * n_plots, 4), layout="tight", dpi=400, squeeze=False
    )
    axs = axs.squeeze(0)
    for ax, H_path in zip(axs, H_paths):
        H = tr.load(H_path)[idx]
        H_name = os.path.basename(H_path)
        plot_waterfall_og(ax, sr=48000, H=H, n_fft=1024, title=H_name, use_log=use_log)

    return fig


def plot_wavetable(wt: T, title: Optional[str] = None) -> Figure:
    assert wt.ndim == 2
    n_pos = wt.size(0)
    # Plot as a column of subplots
    fig, axs = plt.subplots(n_pos, ncols=1, figsize=(8, 4 * n_pos), squeeze=False)
    for idx in range(n_pos):
        ax = axs[idx, 0]
        ax.plot(wt[idx, :].numpy())
        ax.set_title(f"Position {idx}")
        ax.set_ylim(-1.1, 1.1)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    H_paths = [
        os.path.join(OUT_DIR, "coeff_sr__H.pt"),
        os.path.join(OUT_DIR, "coeff_fs_128__H.pt"),
        os.path.join(OUT_DIR, "coeff_fs_512__H.pt"),
        os.path.join(OUT_DIR, "coeff_fs_1024__H.pt"),
        os.path.join(OUT_DIR, "coeff_fs_4096__H.pt"),
        # os.path.join(OUT_DIR, "lp_sr__H.pt"),
        # os.path.join(OUT_DIR, "lp_fs_128__H.pt"),
        # os.path.join(OUT_DIR, "lp_fs_1024__H.pt"),
    ]
    for idx in tqdm(range(68)):
        fig = plot_waterfalls(H_paths, idx, use_log=True)
        fig.savefig(os.path.join(OUT_DIR, f"waterfalls_{idx}.png"))
