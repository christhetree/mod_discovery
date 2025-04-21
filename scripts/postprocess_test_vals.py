import logging
import os

import pandas as pd
from pandas import DataFrame

from paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def extract_test_vals(
    name: str, tsv_path: str, trial_col: str = "seed", x_col: str = "step"
) -> DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", index_col=False)
    df = df[df["stage"] == "test"]
    df = df.drop(columns=["wt_name", "stage", "global_n"])
    df = df.groupby([trial_col, x_col]).mean().reset_index()
    df = df.drop(columns=[trial_col, x_col])
    df = pd.DataFrame(df.mean()).T
    df.insert(0, "name", name)
    return df


if __name__ == "__main__":
    tsv_names_and_paths = [
        ("spline", os.path.join(OUT_DIR, f"out_curr/serum/mss__s24d3D__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        ("8_hz", os.path.join(OUT_DIR, f"out_curr/serum/mss__frame_8_hz__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        ("frame", os.path.join(OUT_DIR, f"out_curr/serum/mss__frame__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        # ("rand_adapt", os.path.join(OUT_DIR, f"out_curr/serum/mss__s24d3D_rand_adapt__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        # ("shan", os.path.join(OUT_DIR, f"out_curr/serum/mss__shan_s24d3D__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        # ("shan_8_hz", os.path.join(OUT_DIR, f"out_curr/serum/mss__shan_frame_8_hz__sm_16_1024__serum__BA_both_lfo_10.tsv")),
        # ("shan_frame", os.path.join(OUT_DIR, f"out_curr/serum/mss__shan_frame__sm_16_1024__serum__BA_both_lfo_10.tsv")),

        # ("spline", os.path.join(OUT_DIR, f"out_curr/mss__s24d3D__lfo__lfo__ase__fm_fold.tsv")),
        # ("spline_nn", os.path.join(OUT_DIR, f"out_curr/mss__s24d3D_nn__lfo__lfo__ase__fm_fold.tsv")),
        # ("cf_8", os.path.join(OUT_DIR, f"out_curr/mss__frame_8_hz__lfo__lfo__ase__fm_fold.tsv")),
        # ("frame", os.path.join(OUT_DIR, f"out_curr/mss__frame__lfo__lfo__ase__fm_fold.tsv")),

        # ("spline", os.path.join(OUT_DIR, f"out_curr/mss__s24d3D__sm_16_1024__ase__ableton_13.tsv")),
        # ("cf_8", os.path.join(OUT_DIR, f"out_curr/mss__frame_8_hz__sm_16_1024__ase__ableton_13.tsv")),
        # ("frame", os.path.join(OUT_DIR, f"out_curr/mss__frame__sm_16_1024__ase__ableton_13.tsv")),
        # ("rand", os.path.join(OUT_DIR, f"out_curr/mss__s24d3D_rand__sm__ase__ableton_13.tsv")),
    ]

    test_df_s = []
    for name, tsv_path in tsv_names_and_paths:
        test_df = extract_test_vals(name, tsv_path)
        test_df_s.append(test_df)

    name_col = "name"
    df = pd.concat(test_df_s).reset_index(drop=True)
    rows = []
    for col in df.columns:
        if col == name_col:
            continue
        sort_ascending = True
        if "pcc" in col:
            sort_ascending = False
        data = df[[name_col, col]].sort_values([col], ascending=sort_ascending)
        sorted_names = data[name_col].tolist()
        sorted_names = [col] + sorted_names
        rows.append(sorted_names)
        sorted_vals = data[col].tolist()
        sorted_vals = [col] + sorted_vals
        rows.append(sorted_vals)

    sorted_cols = ["name"] + list(range(len(tsv_names_and_paths)))
    df = DataFrame(rows, columns=sorted_cols)
    # filter our rows where the name contains "_inv" or "_inv_all"
    # df = df[~df["name"].str.contains("audio__")]
    # df = df[~df["name"].str.contains("_inv")]
    # df = df[~df["name"].str.contains("_inv_all")]
    # df = df[~df["name"].str.contains("_d2")]
    # df = df[~df["name"].str.contains("__ent")]
    # df = df[~df["name"].str.contains("__spec_ent")]
    # df = df[~df["name"].str.contains("__tv")]
    # df = df[~df["name"].str.contains("__tp")]
    # df = df[~df["name"].str.contains("__range_mean")]
    # df = df[~df["name"].str.contains("__min_val")]
    # df = df[~df["name"].str.contains("__max_val")]
    # df = df[~df["name"].str.contains("fad__")]
    # df = df[df["name"].str.contains("_inv")]
    # df = df[df["name"].str.contains("audio__")]
    print(df.to_string())
    print(f"len(df) = {len(df)}")

    # df = df[df[0] == "spline"]
    # df = df[df[0] == "cf_8"]
    # df = df[df[0] == "frame"]
    # df = df[df[0] == "rand"]
    # df = df[df[2] == "frame"]
    df = df[(df[0] == "spline") | (df[1] == "spline")]
    print(df.to_string())
    print(f"len(df) = {len(df)}")
