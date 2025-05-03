import logging
import os
from collections import defaultdict
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Subplot
from pandas import DataFrame
from tqdm import tqdm

from paths import OUT_DIR, WAVETABLES_DIR
from wavetables import BAD_ABLETON_WTS, CONTINUOUS_ABLETON_WTS

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def calc_tv(df: DataFrame, x_col: str, y_col: str) -> (float, float):
    # Check that x_col is monotonically increasing and unique
    assert df[x_col].is_monotonic_increasing
    assert df[x_col].is_unique
    n = len(df)
    y_vals = df[y_col].values
    tv = 0.0
    for i in range(1, n):
        tv += np.abs(y_vals[i] - y_vals[i - 1])
    tv_x_normed = tv / n
    y_min = df[y_col].min()
    y_max = df[y_col].max()
    y_range = y_max - y_min
    y_vals_0to1 = (y_vals - y_min) / y_range
    tv = 0.0
    for i in range(1, n):
        tv += np.abs(y_vals_0to1[i] - y_vals_0to1[i - 1])
    tv_xy_normed = tv / n
    return tv_x_normed, tv_xy_normed


def prepare_tsv_data(
    tsv_path: str,
    stage: str,
    x_col: str,
    y_col: str,
    y_converge_val: float = 0.1,
    trial_col: str = "seed",
    filter_col: Optional[str] = None,
    filter_vals: Optional[List[str]] = None,
    allow_var_n: bool = False,
) -> Dict[str, np.ndarray]:
    tsv_col_names = ["stage", "x_col", "y_col"]
    print_tsv_vals = [stage, x_col, y_col]
    df = pd.read_csv(tsv_path, sep="\t", index_col=False)

    # Filter out rows
    if filter_vals is not None:
        assert filter_col is not None
        if len(filter_vals) == 1:
            tsv_col_names = ["filter_col", "filter_val"] + tsv_col_names
            print_tsv_vals = [filter_col, filter_vals[0]] + print_tsv_vals
        log.debug(f"Number of rows before filtering: {len(df)}")
        df = df[df[filter_col].isin(filter_vals)]
        log.debug(f"Number of rows after filtering: {len(df)}")
    if len(df) == 0:
        return {}

    # df = df[df["seed"] == 1]

    # Filter out stage
    df = df[df["stage"] == stage]
    log.debug(f"Number of rows before removing warmup steps: {len(df)}")
    # Remove sanity check rows
    df = df[~((df["step"] == 0) & (df["stage"] == "val"))]
    log.debug(f"Number of rows after  removing warmup steps: {len(df)}")

    data = defaultdict(list)
    grouped = df.groupby(trial_col)
    n = len(grouped)
    log.debug(f"Number of trials: {n}")
    tsv_col_names.append("n_trials")
    print_tsv_vals.append(n)

    x_val_mins = []
    x_val_maxs = []
    x_val_ranges = []
    # tvs_x_normed = []
    # tvs_xy_normed = []
    # converged = []
    # converged_x_vals = []

    for _, group in grouped:
        # Calc ranges and duration per step
        x_val_min = group[x_col].min()
        x_val_max = group[x_col].max()
        x_val_mins.append(x_val_min)
        x_val_maxs.append(x_val_max)
        x_val_ranges.append(x_val_max - x_val_min)
        # Take mean of y values if there are multiple for each x value (e.g. val / test or grad accumulation)
        grouped_x = group.groupby(x_col).agg({y_col: "mean"})
        for x_val, y_val in grouped_x.itertuples():
            data[x_val].append(y_val)
        # if stage != "test":
        #     # Calc TV
        #     grouped_x.reset_index(drop=False, inplace=True)
        #     tv_x_normed, tv_xy_normed = calc_tv(grouped_x, x_col, y_col)
        #     tvs_x_normed.append(tv_x_normed)
        #     tvs_xy_normed.append(tv_xy_normed)
        # Check for convergence
        # y_val_min = grouped_x[y_col].min()
        # if y_val_min <= y_converge_val:
        #     converged.append(1)
        #     if stage != "test":
        #         # Find first y value less than y_converge_val and corresponding x value
        #         assert grouped_x[x_col].is_monotonic_increasing
        #         con_x_val = grouped_x[grouped_x[y_col] <= y_converge_val][x_col].values[
        #             0
        #         ]
        #         converged_x_vals.append(con_x_val)
        # else:
        #     converged.append(0)

    if not allow_var_n:
        if len(set(x_val_mins)) != 1:
            log.debug(f"Found var min x val {x_val_mins}")
            return {}
        if len(set(x_val_maxs)) != 1:
            log.debug(f"Found var max x val {x_val_maxs}")
            return {}
        if len(set(x_val_ranges)) != 1:
            log.debug("Found var range x val")
            return {}

    x_vals = []
    y_means = []
    y_vars = []
    y_stds = []
    y_mins = []
    y_maxs = []
    y_95cis = []
    y_ns = []
    # We use a for loop to handle jagged data
    for x_val in sorted(data):
        x_vals.append(x_val)
        y_vals = data[x_val]
        n = len(y_vals)
        y_mean = np.mean(y_vals)
        y_var = np.var(y_vals)
        y_std = np.std(y_vals)
        y_sem = y_std / np.sqrt(n)
        y_95ci = 1.96 * y_sem
        y_min = np.min(y_vals)
        y_max = np.max(y_vals)
        y_means.append(y_mean)
        y_vars.append(y_var)
        y_stds.append(y_std)
        y_mins.append(y_min)
        y_maxs.append(y_max)
        y_95cis.append(y_95ci)
        y_ns.append(n)
    if not allow_var_n:
        assert len(set(y_ns)) == 1, "Found var no. of trials across different x vals"
    x_vals = np.array(x_vals)
    y_means = np.array(y_means)
    y_95cis = np.array(y_95cis)
    y_mins = np.array(y_mins)
    y_maxs = np.array(y_maxs)

    # Display mean, 95% CI, and range for test stage
    if stage == "test":
        assert len(data) == 1
        # Calc mean, 95% CI, and range
        y_vals = list(data.values())[0]
        y_mean = np.mean(y_vals)
        y_std = np.std(y_vals)
        y_sem = y_std / np.sqrt(len(y_vals))
        y_95ci = 1.96 * y_sem
        y_min = np.min(y_vals)
        y_max = np.max(y_vals)
        print_tsv_vals.extend(
            [y_mean, y_95ci, y_mean - y_95ci, y_mean + y_95ci, y_min, y_max]
        )
    else:
        print_tsv_vals.extend(["n/a"] * 6)
    tsv_col_names.extend(["y_mean", "y_95ci", "y_mean-", "y_mean+", "y_min", "y_max"])

    # Display variance info
    y_std = np.mean(y_stds)
    tsv_col_names.append("y_std")
    print_tsv_vals.append(y_std)
    # y_var = np.mean(y_vars)
    # tsv_col_names.append("y_var")
    # print_tsv_vals.append(y_var)

    # # Display TV information
    # if stage != "test":
    #     tv_x_normed = np.mean(tvs_x_normed)
    #     tv_xy_normed = np.mean(tvs_xy_normed)
    #     log.info(f"TV {y_col} (x normed): {tv_x_normed:.4f}, "
    #              f"TV {y_col} (xy normed): {tv_xy_normed:.4f}")
    #     print_tsv_vals.extend([tv_x_normed, tv_xy_normed])
    # else:
    #     print_tsv_vals.extend(["n/a"] * 2)
    # tsv_col_names.extend(["tv_x_normed", "tv_xy_normed"])

    # Display convergence information
    # con_rate = np.mean(converged)
    # tsv_col_names.extend(["y_con_val", "y_con_rate"])
    # print_tsv_vals.extend([y_converge_val, con_rate])
    # if stage != "test" and con_rate > 0:
    #     con_x_val = np.mean(converged_x_vals)
    #     con_x_std = np.std(converged_x_vals)
    #     con_x_min = np.min(converged_x_vals)
    #     con_x_max = np.max(converged_x_vals)
    #     con_x_sem = con_x_std / np.sqrt(len(converged_x_vals))
    #     con_x_95ci = 1.96 * con_x_sem
    #     print_tsv_vals.extend([con_x_val, con_x_95ci, con_x_val - con_x_95ci, con_x_val + con_x_95ci, con_x_min, con_x_max])
    # else:
    #     print_tsv_vals.extend(["n/a"] * 6)
    # tsv_col_names.extend(["con_x_val", "con_x_95ci", "con_x_val - con_x_95ci", "con_x_val + con_x_95ci", "con_x_min", "con_x_max"])

    # Display duration information
    x_val_min = x_val_mins[0]
    x_val_max = x_val_maxs[0]
    tsv_col_names.extend([f"x_min", f"x_max"])
    print_tsv_vals.extend([x_val_min, x_val_max])

    return {
        "x_vals": x_vals,
        "y_means": y_means,
        "y_95cis": y_95cis,
        "y_mins": y_mins,
        "y_maxs": y_maxs,
        "tsv_col_names": tsv_col_names,
        "tsv_vals": print_tsv_vals,
    }


def plot_xy_vals(
    ax: Subplot,
    data: Dict[str, np.ndarray],
    title: Optional[str] = None,
    plot_95ci: bool = True,
    plot_range: bool = True,
) -> None:
    x_vals = data["x_vals"]
    y_means = data["y_means"]
    y_95cis = data["y_95cis"]
    y_mins = data["y_mins"]
    y_maxs = data["y_maxs"]

    mean_label = "mean"
    if title is not None:
        mean_label = title
    ax.plot(x_vals, y_means, label=mean_label, lw=2)
    if plot_95ci:
        ax.fill_between(
            x_vals,
            y_means - y_95cis,
            y_means + y_95cis,
            alpha=0.4,
        )
    if plot_range:
        ax.fill_between(x_vals, y_mins, y_maxs, color="gray", alpha=0.4)

    # Labels and legend
    ax.set_xlabel(f"{x_col}")
    ax.set_ylabel(f"{y_col}")
    ax.legend()
    ax.grid(True)


if __name__ == "__main__":
    wt_dir = os.path.join(WAVETABLES_DIR, "ableton")
    wt_names = [f[:-3] for f in os.listdir(wt_dir) if f.endswith(".pt")]
    wt_names = sorted(wt_names)

    filtered_wt_names = []
    for wt_name in wt_names:
        if any(wt_name.startswith(n) for n in CONTINUOUS_ABLETON_WTS):
            filtered_wt_names.append(wt_name)
    wt_names = filtered_wt_names
    wt_names = [None]
    filtered_wt_names = None

    if filtered_wt_names is not None:
        log.info(f"{len(filtered_wt_names)} filtered_wt_names: {filtered_wt_names}")

    tsv_names_and_paths = [
        # ("spline_n", os.path.join(OUT_DIR, f"out_curr/lfo/noise/mss__s24d3D__lfo__ase__ableton_13.tsv")),
        # ("8_hz_n", os.path.join(OUT_DIR, f"out_curr/lfo/noise/mss__frame_8_hz__lfo__ase__ableton_13.tsv")),
        # ("frame_n", os.path.join(OUT_DIR, f"out_curr/lfo/noise/mss__frame__lfo__ase__ableton_13.tsv")),
        # ("spline", os.path.join(OUT_DIR, f"out_curr/lfo/mss__s24d3D_nn__lfo__ase__ableton_13.tsv")),
        # ("8_hz", os.path.join(OUT_DIR, f"out_curr/lfo/mss__frame_8_hz_nn__lfo__ase__ableton_13.tsv")),
        # ("frame", os.path.join(OUT_DIR, f"out_curr/lfo/mss__frame_nn__lfo__ase__ableton_13.tsv")),

        # ("frame", os.path.join(OUT_DIR, f"out_curr/sm/mss__frame__sm_16_1024__ase__ableton_13.tsv")),
        # ("8_hz", os.path.join(OUT_DIR, f"out_curr/sm/mss__frame_8_hz__sm_16_1024__ase__ableton_13.tsv")),
        # ("spline", os.path.join(OUT_DIR, f"out_curr/sm/mss__s24d3D__sm_16_1024__ase__ableton_13.tsv")),
        # ("oracle", os.path.join(OUT_DIR, f"out_curr/sm/mss__oracle__sm_16_1024__ase__ableton_13.tsv")),
        # ("rand_sm", os.path.join(OUT_DIR, f"out_curr/sm/mss__s24d3D_rand__sm_16_1024__ase__ableton_13.tsv")),
        # ("rand", os.path.join(OUT_DIR, f"out_curr/sm/mss__s24d3D_rand__sm__ase__ableton_13.tsv")),
        # ("spline_nn", os.path.join(OUT_DIR, f"out_curr/sm/no_noise/mss__s24d3D_nn__sm_16_1024__ase__ableton_13.tsv")),
        # ("8_hz_nn", os.path.join(OUT_DIR, f"out_curr/sm/no_noise/mss__frame_8_hz_nn__sm_16_1024__ase__ableton_13.tsv")),
        # ("frame_nn", os.path.join(OUT_DIR, f"out_curr/sm/no_noise/mss__frame_nn__sm_16_1024__ase__ableton_13.tsv")),

        # ("frame_gran", os.path.join(OUT_DIR, f"out_curr/serum/mss__frame_gran__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        ("frame", os.path.join(OUT_DIR, f"out_curr/serum/mss__frame__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        ("8_hz", os.path.join(OUT_DIR, f"out_curr/serum/mss__frame_8_hz__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        ("spline", os.path.join(OUT_DIR, f"out_curr/serum/mss__s24d3D__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        ("rand_sm", os.path.join(OUT_DIR, f"out_curr/serum/mss__s24d3D_rand__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        # ("spline_gran", os.path.join(OUT_DIR, f"out_curr/serum/mss__s24d3D_gran__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        # ("8_hz_gran", os.path.join(OUT_DIR, f"out_curr/serum/mss__frame_8_hz_gran__sm_16_1024__serum__BA_both_lfo_10.tsv")),

        # ("shan_frame_gran", os.path.join(OUT_DIR, f"out_curr/serum/mss__shan_frame_gran__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        # ("shan_frame", os.path.join(OUT_DIR, f"out_curr/serum/mss__shan_frame__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        # ("shan_8_hz", os.path.join(OUT_DIR, f"out_curr/serum/mss__shan_frame_8_hz__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        # ("shan_spline", os.path.join(OUT_DIR, f"out_curr/serum/mss__shan_s24d3D__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        # ("shan_rand_sm", os.path.join(OUT_DIR, f"out_curr/serum/mss__shan_s24d3D_rand__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        # ("shan_spline_gran", os.path.join(OUT_DIR, f"out_curr/serum/mss__shan_s24d3D_gran__sm_16_1024__serum__BA_both_lfo_10.tsv")),

        # ("ddsp_frame_gran", os.path.join(OUT_DIR, f"out_curr/serum/mss__ddsp_frame_gran__sm__serum__BA_both_lfo_10.tsv")),
        # ("ddsp_frame", os.path.join(OUT_DIR, f"out_curr/serum/mss__ddsp_frame__sm__serum__BA_both_lfo_10.tsv")),
        # ("ddsp_8_hz", os.path.join(OUT_DIR, f"out_curr/serum/mss__ddsp_frame_8_hz__sm__serum__BA_both_lfo_10.tsv")),
        # ("ddsp_spline", os.path.join(OUT_DIR, f"out_curr/serum/mss__ddsp_s24d3D__sm__serum__BA_both_lfo_10.tsv")),
        # ("ddsp_rand_sm", os.path.join(OUT_DIR, f"out_curr/serum/mss__ddsp_s24d3D_rand__sm__serum__BA_both_lfo_10.tsv")),
        # ("ddsp_spline_gran", os.path.join(OUT_DIR, f"out_curr/serum/mss__ddsp_s24d3D_gran__sm__serum__BA_both_lfo_10.tsv")),
    ]
    # stage = "train"
    # stage = "val"
    stage = "test"
    x_col = "step"
    # x_col = "global_n"

    # Mod signal distance ==============================================================
    lfo = "add_lfo"
    # lfo = "sub_lfo"
    # lfo = "env"

    # inv = ""
    inv = "_inv"
    # inv = "_inv_all"

    # dist_fn = "__esr"
    dist_fn = "__l1"
    # dist_fn = "__mse"
    # dist_fn = "__fft"
    # dist_fn = "__pcc"
    # dist_fn = "__dtw"
    # dist_fn = "__cd"
    # dist_fn = "__fd"

    deriv = ""
    # deriv = "_d1"
    # deriv = "_d2"

    # y_col = f"{lfo}{inv}{dist_fn}{deriv}"

    # Mod signal metrics ===============================================================
    lfo = "add_lfo"
    # lfo = "sub_lfo"
    # lfo = "env"

    # hat = ""
    hat = "_hat"

    inv = ""
    # inv = "_inv"
    # inv = "_inv_all"

    # metric = "range_mean"
    # metric = "min_val"
    # metric = "max_val"
    metric = "ent"
    # metric = "spec_ent"
    # metric = "tv"
    # metric = "tp"

    # y_col = f"{lfo}{hat}{inv}__{metric}"

    # Audio distances ==================================================================
    # metric = "mss"
    # metric = "mel_stft"
    metric = "mfcc"
    # metric = "rms"
    # metric = "sc"
    # metric = "sb"
    # metric = "sf"

    dist_fn = ""
    # dist_fn = "__esr"
    # dist_fn = "__l1"
    # dist_fn = "__mse"
    # dist_fn = "__fft"
    # dist_fn = "__pcc"
    # dist_fn = "__dtw"
    # dist_fn = "__cd"
    # dist_fn = "__fd"

    deriv = ""
    # deriv = "_d1"
    # deriv = "_d2"

    suffix = ""
    # suffix = "__cf_8_hz"
    # suffix = "__inv_all"
    # suffix = "__cf_8_hz__inv_all"

    y_col = f"audio__{metric}{dist_fn}{deriv}{suffix}"
    # y_col = "loss"

    # FAD distances ====================================================================
    # y_col = "fad__clap-2023"
    # y_col = "fad__encodec-emb-48k"
    # y_col = "fad__panns-cnn14-32k"
    # y_col = "fad__panns-wavegram-logmel"

    trial_col = "seed"
    # trial_col = "wt_name"

    # allow_var_n = False
    allow_var_n = True

    df_rows = []
    df_cols = []
    for wt_name in tqdm(wt_names):
        # Keep track of number of rows before plotting
        n_df_rows = len(df_rows)
        # Define filter values
        if wt_name is None:
            filter_vals = filtered_wt_names
        else:
            filter_vals = [wt_name]
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(
            f"{y_col}, wt_name: {wt_name}, stage: {stage}, trial_col: {trial_col}"
        )
        for name, tsv_path in tsv_names_and_paths:
            data = prepare_tsv_data(
                tsv_path,
                stage,
                x_col,
                y_col,
                trial_col=trial_col,
                filter_col="wt_name",
                filter_vals=filter_vals,
                allow_var_n=allow_var_n,
            )
            if not data:
                continue
            plot_xy_vals(ax, data, title=name, plot_95ci=True, plot_range=False)
            df_cols = ["name"] + data["tsv_col_names"]
            df_row = [name] + data["tsv_vals"]
            assert len(df_cols) == len(df_row)
            df_rows.append(df_row)

        # Check that something was plotted
        if len(df_rows) == n_df_rows:
            continue
        # Check that all TSVs were plotted
        if len(df_rows) != n_df_rows + len(tsv_names_and_paths):
            continue

        # Only show plot if not test stage
        if stage != "test":
            ax.set_ylim(bottom=None, top=None)
            # ax.set_ylim(bottom=0.0, top=None)
            # ax.set_ylim(bottom=0.0, top=0.12)
            plt.show()
            plt.pause(0.20)

    df = pd.DataFrame(df_rows, columns=df_cols)
    # const_factor = 10
    # const_factor = 100
    # const_factor = 1000
    # df["y_mean"] = df["y_mean"] * const_factor
    # df["y_95ci"] = df["y_95ci"] * const_factor
    pd.set_option("display.float_format", lambda x: "%.2f" % x)
    print(df.to_string(index=False))
