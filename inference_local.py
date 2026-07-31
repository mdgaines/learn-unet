"""
inference_local.py — Run UNet water body inference on a local PlanetScope scene.

Usage
-----
Basic (all experiments):
    python inference_local.py --image path/to/scene.tif --experiments path/to/experiments

Single experiment:
    python inference_local.py --image path/to/scene.tif --experiments path/to/experiments
        --exp-names baseline_indices_only

Multiple experiments:
    python inference_local.py --image path/to/scene.tif --experiments path/to/experiments
        --exp-names baseline_indices_only optionB_bands_and_indices

Skip plot:
    python inference_local.py --image path/to/scene.tif --experiments path/to/experiments
        --no-plot

Custom output directory:
    python inference_local.py --image path/to/scene.tif --experiments path/to/experiments
        --output path/to/output

Full option list:
    python inference_local.py --help
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for CLI use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Experiment registry ────────────────────────────────────────────────────────
# Maps experiment name → 0-based spectral band indices.
# Add new experiments here as you train them.
EXPERIMENT_BANDS = {
    "baseline_indices_only":     [6, 7, 8, 9],
    "optionB_bands_only":        [0, 1, 2, 3, 4, 5],
    "optionB_bands_and_indices": list(range(10)),
}

EXPERIMENT_LABELS = {
    "baseline_indices_only":     "1: Indices only",
    "optionB_bands_only":        "2: Bands only",
    "optionB_bands_and_indices": "3: Bands + Indices",
}

ENCODER_CHANNELS = [16, 32, 64, 128]
CHIP_SIZE        = 256


# ── Model ─────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=2, encoder_chn=None):
        super().__init__()
        if encoder_chn is None:
            encoder_chn = [16, 32, 64, 128]
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        prev = in_channels
        for ch in encoder_chn:
            self.encoders.append(DoubleConv(prev, ch))
            self.pools.append(nn.MaxPool2d(2))
            prev = ch
        self.bottleneck = DoubleConv(prev, prev * 2)
        prev = prev * 2
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(encoder_chn):
            self.upconvs.append(nn.ConvTranspose2d(prev, ch, 2, stride=2))
            self.decoders.append(DoubleConv(ch * 2, ch))
            prev = ch
        self.output_conv = nn.Conv2d(prev, out_channels, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.pad(x, [0, skip.shape[3]-x.shape[3],
                               0, skip.shape[2]-x.shape[2]])
            x = dec(torch.cat([skip, x], dim=1))
        return self.output_conv(x)


# ── Inference utilities ────────────────────────────────────────────────────────

def load_norm_stats(stats_path: Path):
    """
    Load normalisation stats, handling all saved formats:
      - {"mean": array, "std": array}       old flat format
      - {"local": (mean, std), ...}         per-source, use local
      - {"mukherjee": (mean, std), ...}     per-source, no local — fall back
      - {"global": (mean, std)}             global keyed
    Returns (mean, std) as numpy arrays.
    """
    stats = np.load(stats_path, allow_pickle=True).item()
    if "mean" in stats:
        return stats["mean"], stats["std"]
    if "local" in stats:
        return stats["local"]
    if "mukherjee" in stats:
        print(f"  Warning: no 'local' stats in {stats_path.name} — "
              f"using 'mukherjee' stats. Results may be suboptimal.")
        return stats["mukherjee"]
    key = next(iter(stats))
    print(f"  Warning: using stats key '{key}' from {stats_path.name}")
    return stats[key]


def get_chip_offsets(total: int, chip_size: int):
    """Offsets covering full extent including partial final chip."""
    offsets = list(range(0, total - chip_size + 1, chip_size))
    if not offsets or offsets[-1] + chip_size < total:
        offsets.append(total - chip_size)
    return offsets


def predict_scene(img_path: Path, model: nn.Module,
                  spectral_bands: list, mean: np.ndarray, std: np.ndarray,
                  chip_size: int = 256, device=torch.device("cpu"),
                  verbose: bool = True) -> np.ndarray:
    """
    Tile, predict, and stitch a full scene.
    Returns uint8 array: 0=not-water, 1=water, 255=no-data.
    """
    model.eval()

    with rasterio.open(img_path) as src:
        H, W  = src.height, src.width
        ndwi  = src.read(8, masked=False).astype(np.float32)  # band 8 = NDWI
    nodata_mask = np.isnan(ndwi)

    prob_map  = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.uint8)

    row_offsets = get_chip_offsets(H, chip_size)
    col_offsets = get_chip_offsets(W, chip_size)
    total_chips = len(row_offsets) * len(col_offsets)
    chip_n      = 0

    with rasterio.open(img_path) as src:
        for r in row_offsets:
            for c in col_offsets:
                chip_n += 1
                if verbose and chip_n % 20 == 0:
                    print(f"    chip {chip_n}/{total_chips}", end="\r")

                window = Window(c, r, chip_size, chip_size)
                chip   = src.read([b + 1 for b in spectral_bands],
                                  window=window).astype(np.float32)
                chip   = (chip - mean[:, None, None]) / std[:, None, None]
                chip   = np.nan_to_num(chip, nan=0.0, posinf=1.0, neginf=-1.0)
                tensor = torch.from_numpy(chip).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.softmax(model(tensor), dim=1)[0, 1].cpu().numpy()
                prob_map[r:r+chip_size,  c:c+chip_size] += prob
                count_map[r:r+chip_size, c:c+chip_size] += 1

    if verbose:
        print(f"    {total_chips}/{total_chips} chips done")

    pred = (prob_map / np.maximum(count_map, 1) > 0.5).astype(np.uint8)
    pred[nodata_mask] = 255
    return pred


def save_geotiff(pred: np.ndarray, reference_path: Path, output_path: Path):
    """Write prediction as a georeferenced GeoTIFF."""
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", compress="lzw", nodata=255)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred[np.newaxis, ...])
    print(f"  Saved GeoTIFF: {output_path}")


def save_comparison_plot(fc: np.ndarray, predictions: list,
                         scene_name: str, output_path: Path):
    """
    Save a comparison figure: false colour + one column per experiment.
    predictions: list of (display_label, pred_array)
    """
    mask_cmap = mcolors.ListedColormap(["white", "steelblue", "lightgrey"])
    n_cols    = len(predictions) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 8))

    axes[0].imshow(fc)
    axes[0].set_title("False Colour\n(NIR-R-G)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for col_i, (label, pred) in enumerate(predictions, start=1):
        pred_display = np.where(pred == 255, 2, pred)
        water_pct    = (pred == 1).sum() / max((pred != 255).sum(), 1) * 100
        axes[col_i].imshow(pred_display, cmap=mask_cmap, vmin=0, vmax=2,
                           interpolation="nearest")
        axes[col_i].set_title(f"{label}\nwater={water_pct:.1f}%",
                               fontsize=10, fontweight="bold")
        axes[col_i].axis("off")

    legend_elements = [
        Patch(facecolor="white",     edgecolor="grey", label="Not water"),
        Patch(facecolor="steelblue", edgecolor="grey", label="Water"),
        Patch(facecolor="lightgrey", edgecolor="grey", label="No data"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle(scene_name, fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot:    {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UNet water body inference on a local PlanetScope scene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--image", "-i", required=True, type=Path,
        help="Path to the 10-band PlanetScope GeoTIFF"
    )
    parser.add_argument(
        "--experiments", "-e", required=True, type=Path,
        help="Path to the experiments root directory"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory for predictions (default: <image_dir>/predictions)"
    )
    parser.add_argument(
        "--exp-names", nargs="+", default=None,
        choices=list(EXPERIMENT_BANDS.keys()),
        metavar="EXP_NAME",
        help=("Experiment names to run. Defaults to all available. "
              f"Choices: {list(EXPERIMENT_BANDS.keys())}")
    )
    parser.add_argument(
        "--chip-size", type=int, default=CHIP_SIZE,
        help=f"Chip size for tiling (default: {CHIP_SIZE})"
    )
    parser.add_argument(
        "--no-geotiff", action="store_true",
        help="Skip saving GeoTIFF outputs"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip saving comparison plot"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-chip progress output"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    if not args.experiments.exists():
        print(f"Error: experiments dir not found: {args.experiments}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or args.image.parent / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_names = args.exp_names or list(EXPERIMENT_BANDS.keys())
    device    = torch.device("cpu")

    print(f"\nInput : {args.image}")
    print(f"Output: {output_dir}")
    print(f"Exps  : {exp_names}\n")

    # ── Load false colour for plot ────────────────────────────────────────────
    with rasterio.open(args.image) as src:
        full_img = src.read(masked=False).astype(np.float32)
    fc      = full_img[[5, 3, 2]].transpose(1, 2, 0)
    p2, p98 = np.nanpercentile(fc, (2, 98))
    fc      = np.clip((fc - p2) / (p98 - p2 + 1e-6), 0, 1)

    # ── Run experiments ───────────────────────────────────────────────────────
    predictions = []

    for exp_name in exp_names:
        ckpt_dir   = args.experiments / exp_name
        model_path = ckpt_dir / "best_model.pt"
        stats_path = ckpt_dir / "norm_stats.npy"
        label      = EXPERIMENT_LABELS.get(exp_name, exp_name)

        print(f"── {label} ──────────────────────────────")

        if not model_path.exists():
            print(f"  Skipping — best_model.pt not found at {model_path}")
            continue
        if not stats_path.exists():
            print(f"  Skipping — norm_stats.npy not found at {stats_path}")
            continue

        spectral_bands = EXPERIMENT_BANDS[exp_name]
        mean, std      = load_norm_stats(stats_path)
        print(f"  Bands : {spectral_bands}")

        model = UNet(in_channels=len(spectral_bands),
                     encoder_chn=ENCODER_CHANNELS).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        print(f"  Model : loaded ({sum(p.numel() for p in model.parameters()):,} params)")

        pred = predict_scene(
            args.image, model, spectral_bands, mean, std,
            chip_size=args.chip_size, device=device,
            verbose=not args.quiet
        )

        valid_px  = (pred != 255).sum()
        water_pct = (pred == 1).sum() / max(valid_px, 1) * 100
        print(f"  Water : {water_pct:.2f}% of valid pixels")

        if not args.no_geotiff:
            out_name = f"{args.image.stem}_{exp_name}_pred.tif"
            save_geotiff(pred, args.image, output_dir / out_name)

        predictions.append((label, pred))

    # ── Comparison plot ───────────────────────────────────────────────────────
    if predictions and not args.no_plot:
        plot_path = output_dir / f"{args.image.stem}_comparison.png"
        save_comparison_plot(fc, predictions, args.image.stem, plot_path)

    print(f"\nDone — {len(predictions)} experiment(s) complete.")


if __name__ == "__main__":
    main()
