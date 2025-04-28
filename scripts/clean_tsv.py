import logging
import os

import pandas as pd

from paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    # tsv_path = os.path.join(OUT_DIR, "out_curr/lfo/mss__frame_8_hz_nn__lfo__ase__ableton_13.tsv")
    tsv_path = os.path.join(OUT_DIR, "out/mss__ddsp_frame_gran__sm__serum__BA_both_lfo_10.tsv")

    df = pd.read_csv(tsv_path, sep="\t", index_col=False)
    log.info(f"Loaded {len(df)} rows from {tsv_path}")
    df = df[~(df["stage"] == "train")]
    log.info(f"Number of rows after removing train: {len(df)}")
    df = df[~((df["step"] == 0) & (df["stage"] == "val"))]
    log.info(f"Number of rows after removing warmup steps: {len(df)}")

    save_path = f"{tsv_path[:-4]}__cleaned.tsv"
    df.to_csv(save_path, sep="\t", index=False)
