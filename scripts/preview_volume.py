"""Render the three anatomical planes of one subject's MRI as a PNG.

This is the first visual confirmation that the loading and preprocessing
pipeline actually works on real data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from ScanAlzheimer.data.paths import build_image_path, discover_sessions
from ScanAlzheimer.preprocessing.intensity import preprocess_volume
from ScanAlzheimer.preprocessing.volume import AXIS_NAMES, extract_slice, load_volume

DATA_ROOT = Path("data/raw")
OUTPUT_PATH = Path("docs/figures/volume_preview.png")
VARIANT = "t88_masked_gfc"


def main() -> None:
    sessions = discover_sessions(DATA_ROOT)
    if not sessions:
        raise SystemExit(f"No OASIS sessions found under {DATA_ROOT}")

    stem = sorted(sessions)[0]
    path = build_image_path(sessions[stem], stem, VARIANT)

    volume = load_volume(path)
    print(f"Subject:  {stem}")
    print(f"Variant:  {VARIANT}")
    print(f"Shape:    {volume.shape}")
    print(f"Raw range: [{volume.min():.1f}, {volume.max():.1f}]")

    processed = preprocess_volume(volume, scheme="minmax")
    print(f"After preprocessing: [{processed.min():.3f}, {processed.max():.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor="#111318")
    for axis, ax in enumerate(axes):
        index = processed.shape[axis] // 2
        img = extract_slice(processed, axis, index)
        ax.imshow(img.T, cmap="gray", origin="lower")
        ax.set_title(
            f"axis {axis} — {AXIS_NAMES[axis]}  (index {index})",
            color="white",
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle(f"{stem} — {VARIANT}", color="white")
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
