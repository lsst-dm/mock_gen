#!/usr/bin/env python
"""Create a butler repository for the mock generator (needs the LSST Science Pipelines).

The repository registers the requested (tract, patch) data IDs, the LSSTCam ugrizy bands and the generator's dataset
types. No images are ingested: the mock_coadd_gen pipeline writes them. A tract is a catalogue seed; a patch is
noise_digit * 10 + epoch_digit, with epoch digit 0 meaning the 10-year coadd (so patch 11 is noise 0 at 1 year, 15
noise 0 at 5 years, 10 noise 0 at 10 years, 25 noise 2 at 5 years).

    make_mock_butler_repo.py ~/mocks/repo --tracts 8000 9000 9500 --patches 11 15 10 --npix 3000
"""

import argparse
from pathlib import Path

from lsst.mock.gen.butler import make_repo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", type=Path, help="directory of the new repository")
    parser.add_argument("--tracts", type=int, nargs="+", required=True, metavar="SEED",
                        help="tract ids to register; each is a catalogue seed")
    parser.add_argument("--patches", type=int, nargs="+", required=True, metavar="PATCH",
                        help="patch ids to register for every tract (noise_digit * 10 + epoch_digit, 0 -> 10yr)")
    parser.add_argument("--npix", type=int, default=3000,
                        help="side of each mock frame in pixels; must match the coadd task's canvas_npix "
                             "(default: %(default)s)")
    parser.add_argument("--skymap", default="mock", help="name of the skymap to register (default: %(default)s)")
    args = parser.parse_args()
    make_repo(args.repo, args.tracts, args.patches, skymap_name=args.skymap, npix=args.npix)
    print(f"Created {args.repo} with skymap {args.skymap!r}: {len(set(args.tracts))} tracts x "
          f"{len(set(args.patches))} patches ({args.npix} x {args.npix} px each)")


if __name__ == "__main__":
    main()
