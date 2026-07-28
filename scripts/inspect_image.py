"""Throwaway: verify nibabel can read OASIS Analyze 7.5 images and report
their geometry. Not part of the permanent pipeline."""

from pathlib import Path

import nibabel as nib
import numpy as np
from ScanAlzheimer.data.paths import IMAGE_VARIANTS, build_image_path, discover_sessions

DATA_ROOT = Path("data/raw")

sessions = discover_sessions(DATA_ROOT)
print(f"Sessions found on disk: {len(sessions)}")

stem = sorted(sessions)[0]
session_dir = sessions[stem]
print(f"Inspecting: {stem}\n")

for variant in IMAGE_VARIANTS:
    path = build_image_path(session_dir, stem, variant)
    if not path.exists():
        print(f"{variant:<16} MISSING  ({path.name})")
        continue

    img = nib.load(path)
    data = np.asarray(img.dataobj)
    print(f"{variant:<16} shape={img.shape}  dtype={data.dtype}")
    print(f"{'':<16} zooms={img.header.get_zooms()}")
    print(f"{'':<16} range=[{data.min()}, {data.max()}]")
    print()
