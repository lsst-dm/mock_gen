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

Mock coadds and their truth tables are made directly in a butler repository by a
PipelineTask with no input datasets, so they can be built and run with
`pipetask` or bps/htcondor like any other pipeline. There are no intermediate
`.npy` files: each quantum builds a catalogue, renders one coadd, and writes it.

First make an empty repository and its skymap. The skymap has one tract per
catalogue and one patch per survey depth (patch N is the N-year coadd), with
each patch sized to `--npix` (a DP2 patch is 3000 px):

```bash
make_mock_butler_repo.py ~/mocks/repo --n-tracts 5 --max-epoch-year 10 --npix 3000
```

Then run the generator. The catalogue and depth come from the data ID, not
config: the tract id seeds the catalogue and the patch index is the depth in
years. Pick which coadds to build with a data query (no band is named: the repo
registers only the six ugrizy bands):

```bash
# the 5 x 3 grid of this study: catalogues (tracts) 0-4 at 1, 5 and 10 years
pipetask run -b ~/mocks/repo -i skymaps -o mock/gen -p $MOCK_GEN_DIR/pipelines/mock_coadd_gen.yaml \
    -d "skymap='mock' AND patch IN (1, 5, 10)"
```

| Dataset type | Data ID | Contents |
|---|---|---|
| `deepCoadd` | skymap, tract, patch, band | Coadd in nJy (`ExposureF`) with variance, mask, PSF and a TAN WCS; `x_pix`, `y_pix` in `mock_truth` are its pixel coordinates (origin at the patch corner) |
| `mock_truth`, `mock_truth_peaks` | skymap, tract, patch | The truth tables (one row per peak-like component, and the coadd's noise-free peaks); see `mock_lsst_image_generation.truth_peaks` (`ArrowAstropy`) |

Each run also stores `mockCoaddGen_config`, the task config that produced the
coadd, so the repository records how the data was generated. `deepCoadd` is the
input of the pipelines' coadd detection tasks. Because every mock coadd has the
same band data IDs but a different tract/patch, analyse one coadd at a time by
its tract and patch rather than chaining the runs.

## Layout

| Path | Contents |
|---|---|
| `python/lsst/mock/gen/exposures.py` | Read a saved mock coadd and its truth tables into afw exposures and astropy tables |
| `python/lsst/mock/gen/butler.py` | Make a butler repository, its skymap, bands and dataset types, and ingest coadds |
| `python/lsst/mock/gen/generate.py` | `MockCoaddGenTask`: build a catalogue and render one coadd per quantum |
| `bin.src/make_mock_butler_repo.py` | Command: make an empty repository and skymap for the generator |
| `pipelines/mock_coadd_gen.yaml` | The generator pipeline |
| `tests/` | pytest suite (skipped without the stack) |
