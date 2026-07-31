"""Throwaway: verify nibabel can read OASIS Analyze 7.5 images and report
their geometry. Not part of the permanent pipeline."""

from pathlib import Path

import nibabel as nib
import numpy as np

from ScanAlzheimer.data.paths import IMAGE_VARIANTS, discover_sessions, find_image_path

DATA_ROOT = Path("data/raw")

sessions = discover_sessions(DATA_ROOT)
print(f"Sessions found on disk: {len(sessions)}")

stem = sorted(sessions)[0]
session_dir = sessions[stem]
print(f"Inspecting: {stem}\n")

for variant in IMAGE_VARIANTS:
    path = find_image_path(session_dir, stem, variant)
    if path is None:
        print(f"{variant:<16} MISSING")
        continue

    img = nib.load(path)
    data = np.asarray(img.dataobj)
    print(f"{variant:<16} {path.name}")
    print(f"{'':<16} shape={img.shape}  dtype={data.dtype}")
    print(f"{'':<16} range=[{data.min()}, {data.max()}]")
    print()
