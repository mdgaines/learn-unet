import os
import glob
import zipfile
import subprocess
import multiprocessing
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely import wkb
from concurrent.futures import ProcessPoolExecutor, as_completed

import subprocess
import rasterio
from pyproj import CRS
from collections import Counter

import argparse
from cli import parse_args

import time
import glob

def safe_remove(path, max_attempts=8, delay=0.5, backoff=1.5):
    """
    Windows sometimes holds a brief file lock (often AV scanning a just-written
    file) after a GDAL subprocess exits. Retry deletion with exponential backoff
    instead of failing outright. Returns True if removed, False if it never
    cleared within max_attempts (caller can decide whether that's fatal).
    """
    wait = delay
    for attempt in range(max_attempts):
        try:
            os.remove(path)
            return True
        except (PermissionError, OSError) as e:
            if attempt == max_attempts - 1:
                print(f"Warning: could not remove {path} after {max_attempts} attempts: {e}")
                return False
            time.sleep(wait)
            wait *= backoff
    return False

def retry_on_lock(func, *args, max_attempts=8, delay=0.5, backoff=1.5, **kwargs):
    """
    Generic retry wrapper for any operation that might hit a transient Windows
    file lock (commonly antivirus scanning a just-written file). Retries with
    exponential backoff before giving up and re-raising the original error.
    """
    wait = delay
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except (PermissionError, OSError) as e:
            last_exc = e
            if attempt == max_attempts - 1:
                raise
            time.sleep(wait)
            wait *= backoff
    raise last_exc  # pragma: no cover — unreachable, satisfies linters

# ---------------------------------------------------------------
# STEP 1: Resolve DEM zip files to /vsizip/ paths (no extraction needed)
# ---------------------------------------------------------------
def get_vsizip_paths(dem_zip_dir, pattern="*.zip"):
    zip_files = sorted(glob.glob(os.path.join(dem_zip_dir, pattern)))
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in {dem_zip_dir}")

    vsizip_paths = []
    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path, "r") as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff", ".img"))]
            if not tif_names:
                print(f"Warning: no raster found inside {zip_path}, skipping")
                continue
            if len(tif_names) > 1:
                print(f"Warning: multiple rasters in {zip_path}, using first: {tif_names[0]}")
            vsizip_paths.append(f"/vsizip/{zip_path}/{tif_names[0]}")

    print(f"Resolved {len(vsizip_paths)} raster paths from {len(zip_files)} zip files")
    return vsizip_paths


# ---------------------------------------------------------------
# STEP 2: Build a VRT mosaic (native resolution/CRS — no reprojection yet)
# ---------------------------------------------------------------
# def build_dem_mosaic_from_zips(dem_zip_dir, vrt_path, pattern="*.zip"):
#     vsizip_paths = get_vsizip_paths(dem_zip_dir, pattern)

#     file_list_path = vrt_path.replace(".vrt", "_filelist.txt")
#     with open(file_list_path, "w") as f:
#         f.write("\n".join(vsizip_paths))

#     subprocess.run([
#         "gdalbuildvrt",
#         "-input_file_list", file_list_path,
#         vrt_path
#     ], check=True)

#     print(f"VRT mosaic written to {vrt_path}")
#     return vrt_path

def get_horizontal_crs(wkt):
    """
    Return the horizontal-only pyproj CRS from a WKT string.
    If the CRS is compound (horizontal + vertical, e.g. NAVD88 height),
    extract just the horizontal component.
    """
    crs = CRS.from_wkt(wkt)
    if crs.is_compound:
        return crs.sub_crs_list[0]  # horizontal component is always first
    return crs


def inspect_dem_source(path):
    """Return (dtype, raw_wkt, horizontal_crs) for a single DEM source."""
    try:
        with rasterio.open(path) as src:
            dtype = src.dtypes[0]
            wkt = src.crs.to_wkt() if src.crs else None
            if wkt is None:
                return None
            horizontal_crs = get_horizontal_crs(wkt)
            return dtype, wkt, horizontal_crs
    except Exception as e:
        print(f"Warning: could not inspect {path}: {e}")
        return None


def normalize_dem_sources(vsizip_paths, work_dir, reference_dtype="float32"):
    """
    Inspect every source DEM. Fix dtype if needed. Fix CRS only if the
    HORIZONTAL component genuinely differs from the reference (using pyproj's
    equivalence check, which correctly ignores float-rounding noise in WKT
    and vertical-datum wrapping) — not a raw string comparison.
    """
    norm_dir = os.path.join(work_dir, "normalized_sources")
    os.makedirs(norm_dir, exist_ok=True)

    info = {}
    for p in vsizip_paths:
        result = inspect_dem_source(p)
        if result is not None:
            info[p] = result

    if not info:
        raise RuntimeError("Could not inspect any DEM source files — check paths/permissions.")

    # Reference = the horizontal CRS that appears most often (by EPSG code where available,
    # falling back to WKT string for ones without a clean EPSG match)
    def crs_key(hcrs):
        epsg = hcrs.to_epsg()
        return epsg if epsg is not None else hcrs.to_wkt()

    key_counts = Counter(crs_key(v[2]) for v in info.values())
    reference_key = key_counts.most_common(1)[0][0]
    reference_crs = next(v[2] for v in info.values() if crs_key(v[2]) == reference_key)
    reference_wkt_full = next(v[1] for v in info.values() if crs_key(v[2]) == reference_key)

    reference_prj_path = os.path.join(work_dir, "reference_crs.prj")
    with open(reference_prj_path, "w") as f:
        f.write(reference_crs.to_wkt())  # horizontal-only WKT, used for all fixes

    normalized_paths = []
    flagged_real = []      # genuine horizontal CRS differences — worth a closer look
    flagged_cosmetic = []  # compound-CRS or float-precision noise — safe, just logged
    dtype_fix_count = 0

    for path, (dtype, wkt, hcrs) in info.items():
        needs_dtype_fix = dtype != reference_dtype

        # Equivalence check is for LOGGING/categorization only (cosmetic vs. real).
        # The fix itself must run on ANY raw WKT difference, since gdalbuildvrt
        # does its own strict comparison and doesn't care that pyproj considers
        # them equivalent.
        horizontally_equivalent = hcrs.equals(reference_crs)
        needs_crs_fix = wkt != reference_wkt_full  # <-- fixed: raw WKT difference, not equivalence

        if not needs_dtype_fix and not needs_crs_fix:
            normalized_paths.append(path)
            continue

        safe_name = (os.path.basename(path)
                     .replace(os.sep, "_").replace("/", "_").replace(".tif", ""))
        out_vrt = os.path.join(norm_dir, f"{safe_name}_norm.vrt")

        translate_cmd = ["gdal_translate", "-of", "VRT"]
        if needs_dtype_fix:
            translate_cmd += ["-ot", reference_dtype.capitalize()]
            dtype_fix_count += 1
        if needs_crs_fix:
            translate_cmd += ["-a_srs", reference_prj_path]
            # Categorize for logging based on whether it was a real horizontal
            # difference or just compound-CRS wrapping / float noise
            if not horizontally_equivalent:
                flagged_real.append((path, wkt, reference_wkt_full))
            else:
                flagged_cosmetic.append((path, "compound CRS or float-precision WKT noise — horizontal component confirmed equivalent"))
        translate_cmd += [path, out_vrt]

        result = subprocess.run(translate_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: failed to normalize {path}: {result.stderr.strip()}")
            continue

        normalized_paths.append(out_vrt)

    print(f"\nNormalized {len(normalized_paths)} sources "
          f"({len(flagged_cosmetic)} compound-CRS fixes, "
          f"{len(flagged_real)} unexplained CRS fixes, "
          f"{dtype_fix_count} dtype fixes)")

    if flagged_cosmetic:
        print(f"\n{len(flagged_cosmetic)} file(s) had a compound CRS (horizontal + vertical NAVD88 "
              f"height) — horizontal component confirmed equivalent to reference. Vertical component "
              f"stripped; horizontal coordinates unaffected.")
        for path, note in flagged_cosmetic:
            print(f"  {path}")

    if flagged_real:
        print(f"\n{'!'*60}")
        print(f"WARNING: {len(flagged_real)} file(s) had a horizontal CRS difference NOT explained")
        print("by compound-CRS wrapping. Review these before trusting the force-fix:")
        print(f"{'!'*60}")
        for path, wkt, ref in flagged_real:
            print(f"\n  File: {path}")
            print(f"  Source WKT: {wkt[:200]}...")
            print(f"  Reference WKT: {ref[:200]}...")

    return normalized_paths


def build_dem_mosaic_from_zips(dem_zip_dir, vrt_path, pattern="*.zip", work_dir=None):
    vsizip_paths = get_vsizip_paths(dem_zip_dir, pattern)

    if work_dir is None:
        work_dir = os.path.dirname(vrt_path)

    normalized_paths = normalize_dem_sources(vsizip_paths, work_dir)

    file_list_path = vrt_path.replace(".vrt", "_filelist.txt")
    with open(file_list_path, "w") as f:
        f.write("\n".join(normalized_paths))

    subprocess.run([
        "gdalbuildvrt",
        "-input_file_list", file_list_path,
        vrt_path
    ], check=True)

    print(f"VRT mosaic written to {vrt_path}")
    return vrt_path

# ---------------------------------------------------------------
# STEP 3: Worker function — processes ONE tile (runs in its own process)
# ---------------------------------------------------------------
def process_single_tile(task):
    tile_id = task["tile_id"]
    geom = wkb.loads(task["geom_wkb"])
    vrt_path = task["vrt_path"]
    output_dir = task["output_dir"]
    target_crs = task["target_crs"]
    target_res = task["target_res"]
    analysis_buffer = task["analysis_buffer"]
    slope_edge_buffer = task["slope_edge_buffer"]
    resampling = task["resampling"]
    slope_units = task["slope_units"]

    total_buffer = analysis_buffer + slope_edge_buffer

    try:
        padded_geom = geom.buffer(total_buffer)
        minx, miny, maxx, maxy = padded_geom.bounds

        tmp_dem = os.path.join(output_dir, f"_tmp_dem_{tile_id}.tif")
        tmp_slope = os.path.join(output_dir, f"_tmp_slope_{tile_id}.tif")

        result = subprocess.run([
            "gdalwarp",
            "-t_srs", target_crs,
            "-tr", str(target_res), str(target_res),
            "-te", str(minx), str(miny), str(maxx), str(maxy),
            "-r", resampling,
            "-co", "COMPRESS=LZW",
            "-co", "TILED=YES",
            "-overwrite",
            vrt_path, tmp_dem
        ], capture_output=True, text=True)

        if result.returncode != 0:
            return (tile_id, "skipped", f"warp failed: {result.stderr.strip()}")

        # --- retry-wrapped validity check ---
        def check_valid():
            with rasterio.open(tmp_dem) as chk:
                sample = chk.read(1, masked=True)
                return sample.mask.all()

        try:
            all_nodata = retry_on_lock(check_valid)
        except (PermissionError, OSError) as e:
            return (tile_id, "error", f"could not read tmp_dem after retries: {e}")

        if all_nodata:
            safe_remove(tmp_dem)
            return (tile_id, "skipped", "no valid DEM coverage")

        # --- slope calc (retry the subprocess call itself, in case tmp_dem is still locked) ---
        slope_args = ["gdaldem", "slope", tmp_dem, tmp_slope,
                      "-co", "COMPRESS=LZW", "-co", "TILED=YES"]
        if slope_units == "percent":
            slope_args.append("-p")

        def run_slope():
            r = subprocess.run(slope_args, capture_output=True, text=True)
            if r.returncode != 0:
                raise PermissionError(r.stderr.strip())  # unify retry path for lock-like failures
            return r

        try:
            retry_on_lock(run_slope)
        except Exception as e:
            safe_remove(tmp_dem)
            return (tile_id, "error", f"slope failed after retries: {e}")

        # --- clip DEM + slope, retry-wrapped opens ---
        final_geom = [geom.buffer(analysis_buffer)]

        for src_path, prefix in [(tmp_dem, "dem"), (tmp_slope, "slope")]:
            def do_clip(src_path=src_path, prefix=prefix):
                with rasterio.open(src_path) as src:
                    out_image, out_transform = rio_mask(
                        src, final_geom, crop=True, all_touched=True, filled=True, nodata=src.nodata
                    )
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "compress": "LZW"
                    })
                    out_path = os.path.join(output_dir, f"{prefix}_{tile_id}.tif")
                    with rasterio.open(out_path, "w", **out_meta) as dst:
                        dst.write(out_image)

            retry_on_lock(do_clip)

        safe_remove(tmp_dem)
        safe_remove(tmp_slope)
        return (tile_id, "written", None)

    except Exception as e:
        return (tile_id, "error", str(e))

# ---------------------------------------------------------------
# STEP 4: Tile selection helpers
# ---------------------------------------------------------------
def select_test_tiles(gdf, n=3):
    """
    Select n tiles that are spatially adjacent/overlapping with each other,
    rather than just the first n rows. Adjacent tiles are what actually
    exercise buffer/edge behavior at shared boundaries, which is the main
    thing worth checking before committing to a full run.

    Falls back to the first n tiles if fewer than n adjacent tiles are found
    (e.g. a very sparse or oddly shaped tile grid).
    """
    if len(gdf) <= n:
        return gdf

    sindex = gdf.sindex

    for idx, row in gdf.iterrows():
        # Find tiles whose bounding boxes touch/overlap this one's
        candidates = list(sindex.intersection(row.geometry.buffer(1).bounds))
        candidates = [c for c in candidates if c != idx]
        if len(candidates) >= n - 1:
            selected_idx = [idx] + candidates[: n - 1]
            print(f"Test mode: selected {len(selected_idx)} spatially adjacent tiles: {selected_idx}")
            return gdf.loc[selected_idx]

    print(f"Warning: could not find {n} adjacent tiles, falling back to first {n}")
    return gdf.iloc[:n]


# ---------------------------------------------------------------
# STEP 5: Dispatch all tiles across worker processes
# ---------------------------------------------------------------
def process_dem_tiles(vrt_path, tile_shapefile, output_dir,
                       target_crs="EPSG:32119", target_res=3,
                       analysis_buffer=5, slope_edge_buffer=15,
                       resampling="bilinear", slope_units="degree",
                       id_field=None, max_workers=None,
                       mode="all", num_test_tiles=3):
    """
    mode: 'test' processes only num_test_tiles spatially adjacent tiles
          (for quickly validating buffer/edge behavior before a full run).
          'all' processes every tile in the shapefile.
    """
    os.makedirs(output_dir, exist_ok=True)

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    gdf = gpd.read_file(tile_shapefile)
    if gdf.crs.is_geographic:
        raise ValueError("Tile shapefile CRS is geographic — reproject to a projected CRS before buffering in meters.")

    gdf_target = gdf.to_crs(target_crs)

    if mode == "test":
        gdf_target = select_test_tiles(gdf_target, n=num_test_tiles)
        print(f"TEST MODE: processing {len(gdf_target)} tiles only")
    else:
        print(f"FULL RUN: processing all {len(gdf_target)} tiles")

    tasks = []
    for idx, row in gdf_target.iterrows():
        tile_id = row[id_field] if id_field else idx
        if isinstance(tile_id, float) and tile_id.is_integer():
            tile_id = int(tile_id)

        out_path_slope = os.path.join(output_dir, f"slope_tiles", f"slope_{tile_id}.tif")
        out_path_dem = os.path.join(output_dir, f"dem_tiles", f"dem_{tile_id}.tif")
        if os.path.exists(out_path_slope) and os.path.exists(out_path_dem):
            print(f"{os.path.basename(out_path_slope)} exists. Continuing.")
            continue

        tasks.append({
            "tile_id": tile_id,
            "geom_wkb": row.geometry.wkb,
            "vrt_path": vrt_path,
            "output_dir": output_dir,
            "target_crs": target_crs,
            "target_res": target_res,
            "analysis_buffer": analysis_buffer,
            "slope_edge_buffer": slope_edge_buffer,
            "resampling": resampling,
            "slope_units": slope_units,
        })

    print(f"Processing {len(tasks)} tiles across {max_workers} worker processes...")

    written, skipped, errored = 0, 0, 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_tile, task): task["tile_id"] for task in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            tile_id = futures[future]
            try:
                _, status, msg = future.result()
            except Exception as e:
                status, msg = "error", str(e)

            if status == "written":
                written += 1
            elif status == "skipped":
                skipped += 1
                print(f"[tile {tile_id}] skipped: {msg}")
            else:
                errored += 1
                print(f"[tile {tile_id}] ERROR: {msg}")

            if i % 50 == 0 or i == len(tasks):
                print(f"  progress: {i}/{len(tasks)} tiles processed")

    # Safety-net sweep: remove any orphaned temp files that survived retries
    leftover = glob.glob(os.path.join(output_dir, "_tmp_dem_*.tif")) + \
               glob.glob(os.path.join(output_dir, "_tmp_slope_*.tif"))
    if leftover:
        print(f"\nCleaning up {len(leftover)} leftover temp file(s) from earlier lock contention...")
        for f in leftover:
            safe_remove(f, max_attempts=3, delay=1.0)

    print(f"Done. Wrote {written} tile pairs, skipped {skipped}, errored {errored}.")
    return written, skipped, errored




def main():
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.output_dir + '/slope_tiles/', exist_ok=True)
    os.makedirs(args.output_dir + '/dem_tiles/', exist_ok=True)

    vrt_path = os.path.join(args.work_dir, "dem_mosaic.vrt")

    if args.skip_mosaic:
        if not os.path.exists(vrt_path):
            raise FileNotFoundError(f"--skip-mosaic set but no VRT found at {vrt_path}")
        print(f"Skipping mosaic build, reusing existing VRT: {vrt_path}")
    else:
        build_dem_mosaic_from_zips(args.dem_zip_dir, vrt_path, pattern=args.zip_pattern)

    process_dem_tiles(
        vrt_path=vrt_path,
        tile_shapefile=args.tile_shapefile,
        output_dir=args.output_dir,
        target_crs=args.target_crs,
        target_res=args.target_res,
        analysis_buffer=args.analysis_buffer,
        slope_edge_buffer=args.slope_edge_buffer,
        resampling=args.resampling,
        slope_units=args.slope_units,
        id_field=args.id_field,
        max_workers=args.max_workers,
        mode=args.mode,
        num_test_tiles=args.num_test_tiles,
    )


if __name__ == "__main__":
    main()