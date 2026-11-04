"""PipelineTasks that generate mock catalogues and render them into coadds, with no input datasets.

The pipeline has two tasks. `MockCatalogGenTask` builds one mock catalogue per tract and writes its four tables.
`MockCoaddGenTask` reads a tract's catalogue and renders it into a single-depth ``deepCoadd`` (six bands) plus the
``mock_truth`` and ``mock_truth_peaks`` tables, so one catalogue can be rendered at many depths and noise
realisations without rebuilding it.

The catalogue, depth and noise realisation come from the data ID, not config:

- the tract id is the catalogue seed, so each tract is an independent scene;
- the patch id is ``noise_digit * 10 + epoch_digit``, with ``epoch_digit`` 0 meaning the 10-year coadd (so patch 10
  is noise 0 at 10 years, patch 11 noise 0 at 1 year, patch 23 noise 2 at 3 years). Patches sharing an epoch but
  differing in the noise digit are independent noise realisations of the same scene at the same depth.

Which (tract, patch) data IDs exist is set when the repository is made (`~lsst.mock.gen.butler.make_repo`); a data
query picks which to build, e.g. ``tract IN (8000, 9000) AND patch IN (11, 15, 10)``.
"""

__all__ = [
    "MockCatalogGenConnections", "MockCatalogGenConfig", "MockCatalogGenTask",
    "MockCoaddGenConnections", "MockCoaddGenConfig", "MockCoaddGenTask",
]

import contextlib
import io

import lsst.afw.geom as afwGeom
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from mock_lsst_image_generation.catalogue_pipeline import MockCatalogue, build_mock_catalogue
from mock_lsst_image_generation.coadd_synthesis import MASK_PLANES
from mock_lsst_image_generation.config import IMAGE_CONFIG
from mock_lsst_image_generation.cosmos_catalogue import DEFAULT_COSMOS_PATH, load_cosmos2025_catalogue
from mock_lsst_image_generation.image_pipeline import render_single_coadd
from mock_lsst_image_generation.truth_peaks import find_truth_peaks, truth_table

from .butler import PEAKS_DATASET_TYPE, TRUTH_DATASET_TYPE
from .config import CatalogueConfig, ImageConfig
from .exposures import make_exposure, to_astropy

# The model is calibrated to the 10-year survey, and patch 0 would be a 0-year (no-visit) coadd, so the epoch digit
# 0 is mapped to the 10-year depth rather than rejected.
MAX_EPOCH_YEAR = 10

# The mock frames are built from COSMOS2025, so every tract shares a TAN WCS centred on the COSMOS field. The WCS is
# irrelevant to detection testing; it only gives the exposures a sane sky frame.
_COSMOS_RA = 150.1191
_COSMOS_DEC = 2.2058

_CATALOGUE_TABLES = ("galaxies", "sfregions", "tidal_blobs", "tidal_pairs")


def _cosmos_wcs(npix, pixscale):
    """TAN `~lsst.afw.geom.SkyWcs` centred on COSMOS, with the reference pixel at the frame centre."""
    crpix = geom.Point2D((npix - 1) / 2, (npix - 1) / 2)
    crval = geom.SpherePoint(_COSMOS_RA, _COSMOS_DEC, geom.degrees)
    return afwGeom.makeSkyWcs(crpix, crval, afwGeom.makeCdMatrix(scale=pixscale * geom.arcseconds))


class MockCatalogGenConnections(pipeBase.PipelineTaskConnections, dimensions=("skymap", "tract")):
    """Connections for `MockCatalogGenTask`: the four catalogue tables, no inputs."""

    galaxies = cT.Output(doc="One row per galaxy, star and LSB source.", name="mock_galaxies",
                         storageClass="ArrowAstropy", dimensions=("skymap", "tract"))
    sfregions = cT.Output(doc="Star-forming regions on resolved galaxies.", name="mock_sfregions",
                          storageClass="ArrowAstropy", dimensions=("skymap", "tract"))
    tidal_blobs = cT.Output(doc="Tidal-bridge light blobs between interacting pairs.", name="mock_tidal_blobs",
                            storageClass="ArrowAstropy", dimensions=("skymap", "tract"))
    tidal_pairs = cT.Output(doc="Interacting galaxy pairs linked by a tidal bridge.", name="mock_tidal_pairs",
                            storageClass="ArrowAstropy", dimensions=("skymap", "tract"))


class MockCatalogGenConfig(pipeBase.PipelineTaskConfig, pipelineConnections=MockCatalogGenConnections):
    """Config for `MockCatalogGenTask`. The catalogue seed is the tract id, not here."""

    catalogue = pexConfig.ConfigField(dtype=CatalogueConfig, doc="Scene-content settings.")
    cosmos_path = pexConfig.Field(
        dtype=str, default=None, optional=True,
        doc="COSMOS2025 .npz to build catalogues from; the shipped subset is used when unset.")


class MockCatalogGenTask(pipeBase.PipelineTask):
    """Build one mock catalogue per tract and write its four tables."""

    _DefaultName = "mockCatalogGen"
    ConfigClass = MockCatalogGenConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        seed = butlerQC.quantum.dataId["tract"]
        outputs = self.run(seed=seed)
        for table in _CATALOGUE_TABLES:
            butlerQC.put(getattr(outputs, table), getattr(outputRefs, table))

    def run(self, seed):
        """Build a mock catalogue from the configured settings and the given seed.

        Parameters
        ----------
        seed : `int`
            Seed fixing the realisation (the tract id).

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            ``galaxies``, ``sfregions``, ``tidal_blobs`` and ``tidal_pairs`` as
            `astropy.table.Table`.
        """
        self.log.info("Building mock catalogue (seed %d)", seed)
        cfg = {**self.config.catalogue.build_cfg(), "seed": seed}
        star_cfg = self.config.catalogue.build_star_cfg()
        cosmos = load_cosmos2025_catalogue(self.config.cosmos_path or DEFAULT_COSMOS_PATH)
        # The pipeline captures the generator's output; the model's own progress bars and prints are noise here.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            catalogue = build_mock_catalogue(cosmos, cfg, star_cfg=star_cfg)
        return pipeBase.Struct(**{table: to_astropy(getattr(catalogue, table)) for table in _CATALOGUE_TABLES})


class MockCoaddGenConnections(pipeBase.PipelineTaskConnections, dimensions=("skymap", "tract", "patch"),
                              defaultTemplates={"coaddName": "deep"}):
    """Connections for `MockCoaddGenTask`: a tract's catalogue in, its coadd and truth tables out."""

    galaxies = cT.Input(doc="The tract's galaxy table.", name="mock_galaxies",
                        storageClass="ArrowAstropy", dimensions=("skymap", "tract"))
    sfregions = cT.Input(doc="The tract's star-forming regions.", name="mock_sfregions",
                         storageClass="ArrowAstropy", dimensions=("skymap", "tract"))
    tidal_blobs = cT.Input(doc="The tract's tidal-bridge blobs.", name="mock_tidal_blobs",
                           storageClass="ArrowAstropy", dimensions=("skymap", "tract"))
    tidal_pairs = cT.Input(doc="The tract's interacting pairs.", name="mock_tidal_pairs",
                           storageClass="ArrowAstropy", dimensions=("skymap", "tract"))
    coadds = cT.Output(
        doc="Mock coadd of each band (nJy, with variance, mask, PSF and WCS).",
        name="{coaddName}Coadd", storageClass="ExposureF",
        dimensions=("band", "skymap", "tract", "patch"), multiple=True)
    truth = cT.Output(
        doc="One row per peak-like component of the mock, with per-band flux and peak flags.",
        name=TRUTH_DATASET_TYPE, storageClass="ArrowAstropy", dimensions=("skymap", "tract", "patch"))
    peaks = cT.Output(
        doc="Every noise-free local maximum of the coadd, with the component it belongs to.",
        name=PEAKS_DATASET_TYPE, storageClass="ArrowAstropy", dimensions=("skymap", "tract", "patch"))


class MockCoaddGenConfig(pipeBase.PipelineTaskConfig, pipelineConnections=MockCoaddGenConnections):
    """Config for `MockCoaddGenTask`. The epoch and noise realisation come from the patch id."""

    image = pexConfig.ConfigField(dtype=ImageConfig, doc="Depth, PSF and noise settings.")
    n_workers = pexConfig.Field(dtype=int, default=4, doc="Parallel GalSim rendering processes.")
    min_snr = pexConfig.Field(
        dtype=float, default=1.0, doc="Keep noise-free local maxima with at least this per-pixel S/N.")
    match_radius = pexConfig.Field(
        dtype=float, default=3.0, doc="Pixels within which a maximum belongs to a component.")


class MockCoaddGenTask(pipeBase.PipelineTask):
    """Render one tract's catalogue into a coadd and truth tables per (tract, patch) quantum."""

    _DefaultName = "mockCoaddGen"
    ConfigClass = MockCoaddGenConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        catalogue = MockCatalogue(*(inputs[table].to_pandas() for table in _CATALOGUE_TABLES))
        patch = butlerQC.quantum.dataId["patch"]
        outputs = self.run(catalogue=catalogue, seed=butlerQC.quantum.dataId["tract"], patch=patch,
                           bands=[ref.dataId["band"] for ref in outputRefs.coadds])
        refByBand = {ref.dataId["band"]: ref for ref in outputRefs.coadds}
        for band, exposure in outputs.coadds.items():
            butlerQC.put(exposure, refByBand[band])
        butlerQC.put(outputs.truth, outputRefs.truth)
        butlerQC.put(outputs.peaks, outputRefs.peaks)

    def run(self, catalogue, seed, patch, bands):
        """Render a catalogue into the patch's coadd and build its truth tables.

        Parameters
        ----------
        catalogue : `mock_lsst_image_generation.catalogue_pipeline.MockCatalogue`
            Catalogue to render (the four tables; the diagnostic fields are None).
        seed : `int`
            The tract id; with the patch's noise digit it indexes the render.
        patch : `int`
            Patch id, ``noise_digit * 10 + epoch_digit`` (epoch digit 0 is 10 years).
        bands : `list` [`str`]
            Bands to render, in output order.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            ``coadds`` (`dict` [`str`, `lsst.afw.image.ExposureF`]), ``truth`` and
            ``peaks`` (`astropy.table.Table`).
        """
        noise_digit, epoch_digit = divmod(patch, 10)
        epoch_year = MAX_EPOCH_YEAR if epoch_digit == 0 else epoch_digit
        if not 0 <= noise_digit <= 9:
            raise ValueError(f"patch {patch} out of range: noise digit {noise_digit} must be 0 to 9")
        epoch = f"{epoch_year}y"
        # Vary the render index by the noise digit so same-depth patches are independent noise realisations; the
        # catalogue was built from the tract seed alone, so it is shared across a tract's patches.
        render_index = seed * 10 + noise_digit
        # Merge over IMAGE_CONFIG so render_single_coadd and find_truth_peaks see every key, not just the exposed ones.
        cfg = {**IMAGE_CONFIG, **self.config.image.build_cfg(bands)}

        self.log.info("Rendering the %s coadd (tract %d, patch %d, noise %d)", epoch, seed, patch, noise_digit)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            key, coadd, out_bands, origin, base_image, signal, variance, mask, psf = render_single_coadd(
                catalogue, render_index, cfg=cfg, n_workers=self.config.n_workers, epoch=epoch)

        npix = self.config.image.canvas_npix
        bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(npix, npix))
        wcs = _cosmos_wcs(npix, cfg["pixscale"])
        mask_planes = {plane: 1 << bit for plane, bit in MASK_PLANES.items()}
        coadds = {}
        for index, band in enumerate(out_bands):
            metadata = {"MOCK_COADD": key, "MOCK_PSF_MODEL": coadd.get("psf_model", "gaussian"),
                        "MOCK_N_VISIT": int(coadd["n_visit"][band]), "MOCK_FWHM": float(coadd["fwhm_arcsec"][band])}
            if coadd.get("moffat_beta"):
                metadata["MOCK_MOFFAT_BETA"] = float(coadd["moffat_beta"][band])
            coadds[band] = make_exposure(signal[index], variance[index], mask[index], psf[index], band,
                                         mask_planes, bbox, wcs=wcs, metadata=metadata)

        truth = truth_table(catalogue, out_bands, cfg["pixscale"])
        flagged, found = find_truth_peaks(truth, base_image, origin, coadd, cfg, self.config.min_snr,
                                          self.config.match_radius)
        return pipeBase.Struct(coadds=coadds, truth=to_astropy(flagged), peaks=to_astropy(found))
