"""Create a butler repository for mock coadds and ingest them into it."""

__all__ = [
    "COADD_DATASET_TYPE",
    "PEAKS_DATASET_TYPE",
    "TRUTH_DATASET_TYPE",
    "ingest_mock_coadd",
    "make_dp2like_skymap",
    "make_repo",
    "make_skymap",
    "register_lsstcam_bands",
    "register_skymap",
]

from pathlib import Path

from lsst.daf.butler import Butler, DatasetRef, DatasetType
from lsst.obs.lsst import LsstCam
from lsst.skymap import BaseSkyMap, DiscreteSkyMap
from lsst.utils.introspection import get_full_type_name

from .exposures import make_exposure, read_mock_coadd, read_truth_tables

# The coadd name read by DetectCoaddSourcesTask and MultiBandDetectionTask.
COADD_DATASET_TYPE = "deepCoadd"
TRUTH_DATASET_TYPE = "mock_truth"
PEAKS_DATASET_TYPE = "mock_truth_peaks"

# The mocks are built from COSMOS2025, so the tract sits on the COSMOS field.
_COSMOS_RA = 150.1191
_COSMOS_DEC = 2.2058

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


def make_skymap(
    npix: int, pixel_scale: float = 0.2, ra: float = _COSMOS_RA, dec: float = _COSMOS_DEC
) -> DiscreteSkyMap:
    """Make a skymap of one tract holding one patch of a mock frame.

    Parameters
    ----------
    npix : `int`
        Width and height of the mock frame in pixels.
    pixel_scale : `float`, optional
        Pixel scale in arcsec.
    ra, dec : `float`, optional
        Tract centre in degrees.

    Returns
    -------
    skymap : `lsst.skymap.DiscreteSkyMap`
        Skymap whose tract 0, patch 0 spans pixels ``[0, npix)`` on both
        axes with no border, the frame of a mock rendered with
        ``canvas_npix=npix``.
    """
    config = DiscreteSkyMap.ConfigClass()
    config.raList = [ra]
    config.decList = [dec]
    # A tract wider than one patch would be rounded up to two patches a side.
    config.radiusList = [0.99*npix*pixel_scale/3600/2]
    config.pixelScale = pixel_scale
    config.patchInnerDimensions = [npix, npix]
    config.patchBorder = 0
    config.tractOverlap = 0.0
    config.projection = "TAN"
    return DiscreteSkyMap(config)


def make_dp2like_skymap(
    n_tracts: int, n_patches: int, npix: int = 3000, pixel_scale: float = 0.2,
    ra: float = _COSMOS_RA, dec: float = _COSMOS_DEC
) -> DiscreteSkyMap:
    """Make a skymap for the mock generator: one tract per catalogue, one patch per survey depth.

    Each tract is an independent catalogue realisation and each patch of a tract is one depth (patch N is the N-year
    coadd), so a mock generated for ``(tract, patch)`` is a self-contained ``npix`` x ``npix`` frame. The patches are
    a square grid large enough to hold indices ``0 .. n_patches - 1`` in the first row; the tract overlap and patch
    border are zero, but overlaps would not matter as no coadd uses a neighbour's pixels.

    Parameters
    ----------
    n_tracts : `int`
        Number of tracts (catalogues); tracts are numbered ``0 .. n_tracts - 1``.
    n_patches : `int`
        Number of depth patches to make room for, ``max_epoch_year + 1`` (patch 0 is unused).
    npix : `int`, optional
        Side of each patch, and of each mock frame, in pixels.
    pixel_scale : `float`, optional
        Pixel scale in arcsec.
    ra, dec : `float`, optional
        Centre of the tract strip in degrees; tracts are spread along RA.

    Returns
    -------
    skymap : `lsst.skymap.DiscreteSkyMap`
        Skymap whose patch ``N`` of any tract spans pixels ``[N * npix, (N + 1) * npix)`` in x and ``[0, npix)`` in y.
    """
    # A square tract of just over n_patches patches per axis, so patch indices 0 .. n_patches - 1 exist in the first
    # row. The extra patches above the first row are never addressed (the generator maps patch index to a depth).
    radius = (n_patches * npix - 1) * pixel_scale / 3600 / 2
    config = DiscreteSkyMap.ConfigClass()
    config.raList = [ra + k for k in range(n_tracts)]
    config.decList = [dec] * n_tracts
    config.radiusList = [radius] * n_tracts
    config.pixelScale = pixel_scale
    config.patchInnerDimensions = [npix, npix]
    config.patchBorder = 0
    config.tractOverlap = 0.0
    config.projection = "TAN"
    return DiscreteSkyMap(config)


def register_skymap(butler: Butler, name: str, npix: int, pixel_scale: float = 0.2) -> DiscreteSkyMap:
    """Register a mock frame skymap with a repository.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Writeable butler.
    name : `str`
        Name of the skymap dimension record.
    npix : `int`
        Width and height of the mock frame in pixels.
    pixel_scale : `float`, optional
        Pixel scale in arcsec.

    Returns
    -------
    skymap : `lsst.skymap.DiscreteSkyMap`
        The registered skymap.
    """
    skymap = make_skymap(npix, pixel_scale)
    skymap.register(name, butler)
    return skymap


def make_repo(root: str | Path, n_tracts: int, max_epoch_year: int = 10, npix: int = 3000,
              skymap_name: str = "mock", pixel_scale: float = 0.2) -> Butler:
    """Create and seed a butler repository for the mock coadd generator (`MockCoaddGenTask`).

    The repository is ready to build a no-input quantum graph: its skymap has one tract per catalogue and one depth
    patch per survey year, and the camera, bands and output dataset types are registered. No datasets are ingested;
    the generator writes them. Choose which (tract, patch) coadds to build with a data query, e.g.
    ``patch IN (1, 5, 10)`` over the tracts.

    Parameters
    ----------
    root : `str` or `pathlib.Path`
        Directory of the new repository.
    n_tracts : `int`
        Number of catalogue tracts to register (tracts ``0 .. n_tracts - 1``).
    max_epoch_year : `int`, optional
        Largest depth in years; patches ``0 .. max_epoch_year`` are registered (patch 0 is unused).
    npix : `int`, optional
        Side of each mock frame in pixels; must match the task's ``canvas_npix``.
    skymap_name : `str`, optional
        Name of the skymap to register.
    pixel_scale : `float`, optional
        Pixel scale in arcsec.

    Returns
    -------
    butler : `lsst.daf.butler.Butler`
        Writeable butler for the repository.

    Notes
    -----
    LSSTCam is registered with only its six ugrizy bands (`register_lsstcam_bands`), the ``band`` dimension records
    that coadds are indexed by. ``BaseSkyMap.register`` also writes the skymap into the ``skymaps`` collection, the
    input collection a quantum-graph build needs even though no input datasets are read.
    """
    Butler.makeRepo(str(root))
    butler = Butler.from_config(str(root), writeable=True)
    register_lsstcam_bands(butler)
    skymap = make_dp2like_skymap(n_tracts, max_epoch_year + 1, npix, pixel_scale)
    skymap.register(skymap_name, butler)
    universe = butler.dimensions
    patch = ["skymap", "tract", "patch"]
    for datasetType in (
        DatasetType(COADD_DATASET_TYPE, patch + ["band"], "ExposureF", universe=universe),
        DatasetType(TRUTH_DATASET_TYPE, patch, "ArrowAstropy", universe=universe),
        DatasetType(PEAKS_DATASET_TYPE, patch, "ArrowAstropy", universe=universe),
    ):
        butler.registry.registerDatasetType(datasetType)
    return butler


def ingest_mock_coadd(
    butler: Butler,
    image_dir: str | Path,
    coadd: str | None = None,
    run: str | None = None,
    skymap_name: str = "mock",
) -> list[DatasetRef]:
    """Ingest a mock coadd and its truth tables into a repository.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Writeable butler for a repository made by `make_repo`.
    image_dir : `str` or `pathlib.Path`
        Directory written for one catalogue, ``<out_dir>/<name>``.
    coadd : `str`, optional
        Key of the coadd to ingest. May be omitted when exactly one coadd in
        the directory was saved.
    run : `str`, optional
        Run collection to write to. Defaults to ``mock/<name>/<coadd>``.
    skymap_name : `str`, optional
        Skymap whose tract 0, patch 0 the coadd is assigned to.

    Returns
    -------
    refs : `list` [`lsst.daf.butler.DatasetRef`]
        References to the ingested coadd of each band, then the truth
        tables that exist on disk.

    Raises
    ------
    ValueError
        Raised if the coadd does not lie within the skymap's patch.
    """
    mock = read_mock_coadd(image_dir, coadd)
    skymap = butler.get(
        BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        skymap=skymap_name,
        collections=BaseSkyMap.SKYMAP_RUN_COLLECTION_NAME,
    )
    tract = skymap[0]
    patch = tract[0]
    if not patch.getOuterBBox().contains(mock.bbox):
        raise ValueError(
            f"coadd bbox {mock.bbox} is not within patch {patch.getOuterBBox()} of skymap {skymap_name!r}; "
            "render the mock with canvas_npix matching the skymap"
        )
    if run is None:
        run = f"mock/{mock.name}/{mock.key}"
    butler.registry.registerRun(run)
    dataId = dict(skymap=skymap_name, tract=tract.tract_id, patch=patch.sequential_index)

    # One band at a time: a full frame is ~200 MB per band.
    refs = []
    for band in mock.bands:
        exposure = make_exposure(mock, band, tract.getWcs())
        refs.append(butler.put(exposure, COADD_DATASET_TYPE, dataId, band=band, run=run))
    datasetTypes = {"truth": TRUTH_DATASET_TYPE, "peaks": PEAKS_DATASET_TYPE}
    for kind, table in read_truth_tables(mock).items():
        refs.append(butler.put(table, datasetTypes[kind], dataId, run=run))
    return refs
