import json
import glob
import os

fnames = sorted(glob.glob("/path_to//serum-nsynth//*.wav"))
preset_set = set()

with open("preset_params_v2.json") as f:
    data = json.load(f)

for fname in fnames:
    fname = os.path.basename(fname)
    preset_name_str = fname.split("_")
    if len(preset_name_str) != 2:
        assert len(preset_name_str) == 3, f"Error processing {fname}"
        preset_name = "_".join(preset_name_str[:2])
        pitch, velocity = preset_name_str[2].split("-")
    else:
        preset_name = preset_name_str[0]
        pitch, velocity = preset_name_str[1].split("-")
    
    pitch = int(pitch)
    a_osc_pitch = pitch + data[preset_name]["pitch"]["A Osc"]["pitch_correction"]
    b_osc_pitch = pitch + data[preset_name]["pitch"]["B Osc"]["pitch_correction"]
