"""Create and seed a butler repository for the mock generator (`~lsst.mock.gen.generate`)."""

__all__ = [
    "CATALOGUE_DATASET_TYPES",
    "COADD_DATASET_TYPE",
    "PEAKS_DATASET_TYPE",
    "TRUTH_DATASET_TYPE",
    "make_repo",
    "register_lsstcam_bands",
]

from pathlib import Path

from lsst.daf.butler import Butler, DatasetType
from lsst.obs.lsst import LsstCam
from lsst.utils.introspection import get_full_type_name

from .skymap import MockSkyMap, MockSkyMapConfig

# The coadd name read by the coadd detection tasks (DetectCoaddSourcesTask, MultiResolutionDetectionTask).
COADD_DATASET_TYPE = "deepCoadd"
TRUTH_DATASET_TYPE = "mock_truth"
PEAKS_DATASET_TYPE = "mock_truth_peaks"
# The four tables MockCatalogGenTask writes and MockCoaddGenTask reads back.
CATALOGUE_DATASET_TYPES = ("mock_galaxies", "mock_sfregions", "mock_tidal_blobs", "mock_tidal_pairs")

# The one canonical LSSTCam physical filter per mock band. Registering only these keeps the repo's band dimension to
# the six mock bands, so the generator's quantum graph makes one coadd per band with no band data query; the full
# LsstCam filter set would add engineering bands (grid, spot, ...) and so 25 band data IDs per patch.
_UGRIZY_PHYSICAL = dict(u="u_24", g="g_6", r="r_57", i="i_39", z="z_20", y="y_10")


def register_lsstcam_bands(butler: Butler) -> None:
    """Register the LSSTCam instrument and its six ugrizy bands (no detectors) in a writeable butler.

    Only the ``band`` dimension records are needed to index coadds, and only the six mock bands are registered, so a
    no-input quantum graph over this repository yields exactly those bands. The full ``LsstCam.register`` would also
    add every engineering filter, and so 25 bands.
    """
    instrument = LsstCam()
    translator = instrument.translatorClass
    obs_max = translator.max_exposure_id()
    registry = butler.registry
    with registry.transaction():
        registry.syncDimensionData("instrument", {
            "name": instrument.getName(),
            "detector_max": max(translator.DETECTOR_MAX, 1),
            "visit_max": obs_max,
            "exposure_max": obs_max,
            "class_name": get_full_type_name(instrument),
            "visit_system": None if instrument.visitSystem is None else instrument.visitSystem.value,
        })
        for band, physical in _UGRIZY_PHYSICAL.items():
            registry.syncDimensionData(
                "physical_filter", {"instrument": instrument.getName(), "name": physical, "band": band})


def make_repo(root: str | Path, tracts, patches, skymap_name: str = "mock", npix: int = 3000) -> Butler:
    """Create and seed a butler repository for the mock generator.

    The repository is ready to build a no-input quantum graph: a `MockSkyMap`
    (the requested tracts over one COSMOS frame), the LSSTCam ugrizy bands and
    the generator's dataset types are registered. No datasets are ingested
    except the skymap; the pipeline writes the rest. The tracts and patches
    carry no real sky geometry (a tract is a catalogue seed and a patch encodes
    a noise realisation and a survey epoch; see `~lsst.mock.gen.generate`), but
    the skymap gives coadd tasks that need one (e.g. the legacy
    `MergeDetectionsTask`) a tract to look up.

    Parameters
    ----------
    root : `str` or `pathlib.Path`
        Directory of the new repository.
    tracts : iterable [`int`]
        Tract ids to register; each is a catalogue seed.
    patches : iterable [`int`]
        Patch ids to build for; the skymap patch grid is sized to hold the
        largest. A patch id is ``noise_digit * 10 + epoch``.
    skymap_name : `str`, optional
        Name of the skymap to register.
    npix : `int`, optional
        Side of each mock frame (one patch) in pixels; matched by the coadd
        task's ``canvas_npix``.

    Returns
    -------
    butler : `lsst.daf.butler.Butler`
        Writeable butler for the repository.
    """
    tracts = sorted(set(int(t) for t in tracts))
    patches = sorted(set(int(p) for p in patches))
    Butler.makeRepo(str(root))
    butler = Butler.from_config(str(root), writeable=True)
    register_lsstcam_bands(butler)

    # The skymap registers its own dimension records (skymap, tract, patch) and writes the SkyMap dataset into the
    # skymaps RUN, which is also the non-empty input collection a no-input quantum graph needs.
    config = MockSkyMapConfig()
    config.tractIds = tracts
    config.npix = npix
    config.nPatches = max(patches) + 1
    MockSkyMap(config).register(skymap_name, butler)

    universe = butler.dimensions
    patch_dims = ["skymap", "tract", "patch"]
    tract_dims = ["skymap", "tract"]
    for datasetType in (
        DatasetType(COADD_DATASET_TYPE, patch_dims + ["band"], "ExposureF", universe=universe),
        DatasetType(TRUTH_DATASET_TYPE, patch_dims, "ArrowAstropy", universe=universe),
        DatasetType(PEAKS_DATASET_TYPE, patch_dims, "ArrowAstropy", universe=universe),
        *(DatasetType(name, tract_dims, "ArrowAstropy", universe=universe) for name in CATALOGUE_DATASET_TYPES),
    ):
        butler.registry.registerDatasetType(datasetType)
    return butler
