# mock_gen

LSST Science Pipelines wrapper for
[mock_lsst_image_generation](https://github.com/lsst-dm/mock_lsst_image_generation),
importable as `lsst.mock.gen`. It turns the mock catalogues and coadds that
package renders into Science Pipelines data products: afw exposures, a butler
repository with its skymap and dataset types, and a `MockCoaddGenTask`
PipelineTask that writes mock coadds and their truth tables straight to a
butler.

The mock generator itself (the physical models, the GalSim rendering, the truth
tables) lives in `mock_lsst_image_generation` and depends only on numpy, scipy,
pandas and GalSim. This package is where the stack dependency lives.

## Requirements

- An LSST Science Pipelines stack (`lsst_distrib`).
- `mock_lsst_image_generation` on `PYTHONPATH`. It is not an EUPS product: clone
  it and add its repository root to `PYTHONPATH`, e.g.

  ```bash
  export PYTHONPATH="/path/to/mock_lsst_image_generation:$PYTHONPATH"
  ```

With a stack loaded, set this package up and build it from its root:

```bash
setup -j -r .
scons
```

`scons` writes the command in `bin.src/` to `bin/` (on `PATH` once set up) and
runs the tests.

## Butler repository and the generator pipeline

Mock catalogues and coadds are made directly in a butler repository by a
two-task pipeline with no input datasets, so they can be built and run with
`pipetask` or bps/htcondor like any other pipeline. `mockCatalogGen` builds one
catalogue per tract; `mockCoaddGen` renders a tract's catalogue into a
single-depth coadd and its truth tables, so one catalogue can be rendered at
many depths and noise realisations without rebuilding it.

The catalogue, depth and noise realisation come from the data ID, not config:

- `tract` is the catalogue seed, so each tract is an independent scene;
- `patch` is `noise_digit * 10 + epoch_digit`, with `epoch_digit` 0 meaning the
  10-year coadd (so patch 10 is noise 0 at 10 years, patch 11 noise 0 at 1 year,
  patch 25 noise 2 at 5 years). Patches sharing an epoch but differing in the
  noise digit are independent noise realisations of the same scene at the same
  depth.

First make an empty repository, registering the tract and patch data IDs to
build. The tracts and patches carry no sky geometry (each coadd gets a fixed
COSMOS TAN WCS); `--npix` is the frame size and must match the coadd task's
`canvas_npix` (a DP2 patch is 3000 px):

```bash
make_mock_butler_repo.py ~/mocks/repo --tracts 8000 9000 9500 --patches 11 15 10 --npix 3000
```

Then run the generator. Pick which (tract, patch) to build with a data query (no
band is named: the repo registers only the six ugrizy bands):

```bash
# tracts 8000 and 9000, each at 1, 5 and 10 years
pipetask run -b ~/mocks/repo -i skymaps -o mock/gen -p $MOCK_GEN_DIR/pipelines/mock_coadd_gen.yaml \
    -d "skymap='mock' AND tract IN (8000, 9000) AND patch IN (11, 15, 10)"
```

| Dataset type | Data ID | Contents |
|---|---|---|
| `mock_galaxies`, `mock_sfregions`, `mock_tidal_blobs`, `mock_tidal_pairs` | skymap, tract | The four catalogue tables (`ArrowAstropy`), reused across a tract's patches |
| `deepCoadd` | skymap, tract, patch, band | Coadd in nJy (`ExposureF`) with variance, mask, PSF and a TAN WCS; `x_pix`, `y_pix` in `mock_truth` are its pixel coordinates (origin at the frame corner) |
| `mock_truth`, `mock_truth_peaks` | skymap, tract, patch | The truth tables (one row per peak-like component, and the coadd's noise-free peaks); see `mock_lsst_image_generation.truth_peaks` (`ArrowAstropy`) |

Each run also stores `mockCatalogGen_config` and `mockCoaddGen_config`, the task
configs that produced the data, so the repository records how it was generated.
`deepCoadd` is the input of the pipelines' coadd detection tasks. Because every
mock coadd has the same band data IDs but a different tract/patch, analyse one
coadd at a time by its tract and patch rather than chaining the runs.

## Detection pipelines

Two detection pipelines run on the generated coadds, to compare the new
multi-band detection with the legacy per-band detect-then-merge flow:

```bash
# new: joint multi-band detection, one merged catalogue (object_detection_merged)
pipetask run -b REPO -i <gen run> -o mock/detect \
    -p $MOCK_GEN_DIR/pipelines/detect.yaml -d "<query>"

# legacy: per-band detection then merge (object_detection_merged_legacy). The
# merge needs the skymap, so skymaps is in the input collections.
pipetask run -b REPO -i <gen run>,skymaps -o mock/legacy_detect \
    -p $MOCK_GEN_DIR/pipelines/legacy_detect.yaml -d "<query>"
```

`make_repo` stores a `MockSkyMap` in the `skymaps` collection for the legacy
merge; `detect.yaml` needs only `deepCoadd`.

## Layout

| Path | Contents |
|---|---|
| `python/lsst/mock/gen/config.py` | Nested `pexConfig` classes: the scene, depth, PSF and noise knobs worth varying |
| `python/lsst/mock/gen/generate.py` | `MockCatalogGenTask` and `MockCoaddGenTask` |
| `python/lsst/mock/gen/exposures.py` | Turn rendered arrays into an afw exposure, and a truth table into astropy |
| `python/lsst/mock/gen/butler.py` | Make a butler repository: register the data IDs, bands and dataset types |
| `python/lsst/mock/gen/skymap.py` | `MockSkyMap`: arbitrary tract ids over one COSMOS frame, for tasks that need a skymap |
| `bin.src/make_mock_butler_repo.py` | Command: make an empty repository for the generator |
| `pipelines/mock_coadd_gen.yaml` | The generator pipeline (both tasks) |
| `pipelines/detect.yaml`, `pipelines/legacy_detect.yaml` | Detection pipelines (new multi-band; legacy per-band + merge) |
| `tests/` | pytest suite (skipped without the stack) |
