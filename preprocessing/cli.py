#!/usr/bin/env python
import argparse
import os
# from dem_pipeline import build_dem_mosaic_from_zips, process_dem_tiles


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mosaic zipped 3ft DEM tiles, reproject/resample to target CRS/resolution, "
                    "calculate slope, and clip DEM+slope into buffered tiles matching a tile shapefile."
    )

    # Inputs
    parser.add_argument("--dem-zip-dir", required=True,
                        help="Directory containing zipped DEM tiles")
    parser.add_argument("--zip-pattern", default="*.zip",
                        help="Glob pattern for DEM zip files (default: *.zip)")
    parser.add_argument("--tile-shapefile", required=True,
                        help="Shapefile defining tile footprints to clip against (e.g. tile_grid.shp)")
    parser.add_argument("--id-field", default=None,
                        help="Shapefile attribute field used to name output tiles (default: row index)")

    # Working / output paths
    parser.add_argument("--work-dir", default="./dem_processing",
                        help="Directory for intermediate files (VRT mosaic, file lists)")
    parser.add_argument("--output-dir", default="./output_dem_slope_tiles",
                        help="Directory for final DEM/slope tile outputs")

    # Reprojection / resolution
    parser.add_argument("--target-crs", default="EPSG:32119",
                        help="Target CRS for reprojection (default: EPSG:32119, NC State Plane meters)")
    parser.add_argument("--target-res", type=float, default=3.0,
                        help="Target resolution in target CRS units, e.g. meters (default: 3)")
    parser.add_argument("--resampling", default="bilinear",
                        choices=["nearest", "bilinear", "cubic", "cubicspline", "lanczos", "average"],
                        help="Resampling method for reprojection (default: bilinear)")

    # Buffers
    parser.add_argument("--analysis-buffer", type=float, default=5.0,
                        help="Buffer (in target CRS units) to keep in final output tiles (default: 5)")
    parser.add_argument("--slope-edge-buffer", type=float, default=15.0,
                        help="Extra padding beyond analysis-buffer used only during warp/slope "
                            "calculation, to avoid edge artifacts, then cropped off (default: 15)")

    # Slope
    parser.add_argument("--slope-units", default="degree", choices=["degree", "percent"],
                        help="Units for slope calculation (default: degree)")

    # Parallelism
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Number of parallel worker processes (default: cpu_count - 1)")

    # Run mode
    parser.add_argument("--mode", default="all", choices=["test", "all"],
                        help="'test' processes a small number of spatially adjacent tiles only, "
                            "for validating buffer/edge behavior before a full run. "
                            "'all' processes every tile. (default: all)")
    parser.add_argument("--num-test-tiles", type=int, default=3,
                        help="Number of adjacent tiles to process in test mode (default: 3)")

    # Skip mosaic rebuild if already done
    parser.add_argument("--skip-mosaic", action="store_true",
                        help="Skip VRT mosaic build and reuse existing VRT at <work-dir>/dem_mosaic.vrt")

    return parser.parse_args()


