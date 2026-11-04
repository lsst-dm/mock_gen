#!/usr/bin/env python
"""Create a butler repository for the mock coadd generator (needs the LSST Science Pipelines).

The repository is seeded with a skymap of one tract per catalogue and one patch per survey depth (patch N is the
N-year coadd), the LSSTCam bands, and the mock dataset types. No images are ingested: the MockCoaddGenTask pipeline
writes them. Pick which coadds to build with a data query, e.g. --data-query "patch IN (1, 5, 10)".

    make_mock_butler_repo.py ~/mocks/repo --n-tracts 5 --max-epoch-year 10 --npix 3000
"""

import argparse
from pathlib import Path

from lsst.mock.gen.butler import make_repo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", type=Path, help="directory of the new repository")
    parser.add_argument("--n-tracts", type=int, required=True, help="number of catalogue tracts to register")
    parser.add_argument("--max-epoch-year", type=int, default=10,
                        help="largest depth in years; patches 0..this are registered (default: %(default)s)")
    parser.add_argument("--npix", type=int, default=3000,
                        help="side of each mock frame in pixels; must match the task's canvas_npix (default: "
                             "%(default)s)")
    parser.add_argument("--skymap", default="mock", help="name of the skymap to register (default: %(default)s)")
    parser.add_argument("--pixel-scale", type=float, default=0.2, help="arcsec per pixel (default: %(default)s)")
    args = parser.parse_args()
    make_repo(args.repo, args.n_tracts, max_epoch_year=args.max_epoch_year, npix=args.npix, skymap_name=args.skymap,
              pixel_scale=args.pixel_scale)
    print(f"Created {args.repo} with skymap {args.skymap!r}: {args.n_tracts} tracts x "
          f"{args.max_epoch_year + 1} patches ({args.npix} x {args.npix} px each)")


if __name__ == "__main__":
    main()
