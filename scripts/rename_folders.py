import os
import re

# ==== Configuration ====
root_dir = '/Users/puntland/local_christhetree/aim/acid_ddsp/out/out_curr/lfo/tbd/train__ase__lfo_frame_8_hz 2/acid_ddsp_2'  # <-- Change this to your folder path
N = 65  # The increment constant

# ==== Regex pattern for matching version folders ====
pattern = re.compile(r'^version_(\d+)$')

# ==== First, collect all matching folder names ====
folders = []
for name in os.listdir(root_dir):
    full_path = os.path.join(root_dir, name)
    if os.path.isdir(full_path):
        match = pattern.match(name)
        if match:
            num = int(match.group(1))
            folders.append((num, name))

# ==== Sort to avoid overwrite conflicts ====
folders.sort(reverse=True)  # Rename higher numbers first to avoid conflicts

# ==== Rename each folder ====
for old_num, old_name in folders:
    new_num = old_num + N
    new_name = f'version_{new_num}'
    old_path = os.path.join(root_dir, old_name)
    new_path = os.path.join(root_dir, new_name)
    print(f'Renaming {old_path} -> {new_path}')
    os.rename(old_path, new_path)

print("Renaming complete.")
