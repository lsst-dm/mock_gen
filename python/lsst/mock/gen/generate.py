"""A PipelineTask that generates mock coadds and their truth tables with no input datasets.

Each quantum builds one mock catalogue, renders it into a single coadd at one survey depth, and writes the six-band
``deepCoadd`` plus the ``mock_truth`` and ``mock_truth_peaks`` tables. The catalogue and depth are taken from the
quantum's data ID, not from config:

- the tract id is the catalogue's seed (and the index that seeds the stars and the coadd noise), so each tract is an
  independent realisation;
- the patch index is the depth in whole years, so patch 1 is the 1-year coadd, patch 5 the 5-year, patch 10 the
  10-year; patch 0 and patches past 10 are rejected.

Which (tract, patch) coadds are built is chosen by the registered skymap and the data query, e.g. five tracts with a
data query of ``patch IN (1, 5, 10)`` makes the 5 x 3 grid of this study. The repository and its skymap are made by
``make_mock_butler_repo.py`` (see `~mock_lsst_image_generation.lsst.butler.make_repo`).
"""

__all__ = ["MockCoaddGenConnections", "MockCoaddGenConfig", "MockCoaddGenTask"]

import contextlib
import io

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.skymap import BaseSkyMap

from mock_lsst_image_generation.catalogue_pipeline import build_mock_catalogue
from mock_lsst_image_generation.coadd_synthesis import MASK_PLANES
from mock_lsst_image_generation.config import IMAGE_CONFIG
from mock_lsst_image_generation.cosmos_catalogue import DEFAULT_COSMOS_PATH, load_cosmos2025_catalogue
from mock_lsst_image_generation.image_pipeline import render_single_coadd
from mock_lsst_image_generation.truth_peaks import find_truth_peaks, truth_table

from .butler import PEAKS_DATASET_TYPE, TRUTH_DATASET_TYPE
from .exposures import MockCoadd, make_exposure, to_astropy

# Patch index 0 is unused (a 0-year coadd has no visits) and the model is calibrated to the 10-year survey.
MAX_EPOCH_YEAR = 10


class MockCoaddGenConnections(pipeBase.PipelineTaskConnections, dimensions=("skymap", "tract", "patch"),
                              defaultTemplates={"coaddName": "deep"}):
    """Connections for `MockCoaddGenTask`: a skymap, and the coadd and truth-table outputs."""

    skyMap = cT.Input(
        doc="Skymap whose tract and patch give the coadd's WCS and position.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    coadds = cT.Output(
        doc="Mock coadd of each band (nJy, with variance, mask, PSF and WCS).",
        name="{coaddName}Coadd",
        storageClass="ExposureF",
        dimensions=("band", "skymap", "tract", "patch"),
        multiple=True,
    )
    truth = cT.Output(
        doc="One row per peak-like component of the mock, with per-band flux and peak flags.",
        name=TRUTH_DATASET_TYPE,
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch"),
    )
    peaks = cT.Output(
        doc="Every noise-free local maximum of the coadd, with the component it belongs to.",
        name=PEAKS_DATASET_TYPE,
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch"),
    )


class MockCoaddGenConfig(pipeBase.PipelineTaskConfig, pipelineConnections=MockCoaddGenConnections):
    """Config for `MockCoaddGenTask`. Catalogue and depth are set by the data ID, not here."""

    canvas_npix = pexConfig.Field(
        dtype=int, default=3000,
        doc="Side of each coadd in pixels; must fit the skymap patch (a DP2 patch is 3000).")
    psf_model = pexConfig.ChoiceField(
        dtype=str, default="moffat",
        allowed={"gaussian": "Gaussian core, halo and spikes on stars only",
                 "moffat": "Moffat core with halo and spikes shared by every source"},
        doc="PSF model shared by the coadd's bands.")
    n_workers = pexConfig.Field(
        dtype=int, default=4, doc="Parallel GalSim rendering processes.")
    min_snr = pexConfig.Field(
        dtype=float, default=1.0, doc="Keep noise-free local maxima with at least this per-pixel S/N.")
    match_radius = pexConfig.Field(
        dtype=float, default=3.0, doc="Pixels within which a maximum belongs to a component.")
    cosmos_path = pexConfig.Field(
        dtype=str, default=None, optional=True,
        doc="COSMOS2025 .npz to build catalogues from; the shipped subset is used when unset.")
    config_overrides = pexConfig.DictField(
        keytype=str, itemtype=float, default={},
        doc="Numeric overrides folded into the catalogue CONFIG (e.g. bcg_per_deg2); for structured keys use a "
            "python block in the pipeline file.")
    image_config_overrides = pexConfig.DictField(
        keytype=str, itemtype=float, default={},
        doc="Numeric overrides folded into IMAGE_CONFIG.")


class MockCoaddGenTask(pipeBase.PipelineTask):
    """Generate one mock coadd and its truth tables per (tract, patch) quantum."""

    _DefaultName = "mockCoaddGen"
    ConfigClass = MockCoaddGenConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        skyMap = butlerQC.get(inputRefs.skyMap)
        dataId = butlerQC.quantum.dataId
        tract = skyMap[dataId["tract"]]
        patch = tract[dataId["patch"]]
        outputs = self.run(tract=tract, patch=patch, seed=dataId["tract"], patch_index=dataId["patch"],
                           bands=[ref.dataId["band"] for ref in outputRefs.coadds])
        refByBand = {ref.dataId["band"]: ref for ref in outputRefs.coadds}
        for band, exposure in outputs.coadds.items():
            butlerQC.put(exposure, refByBand[band])
        butlerQC.put(outputs.truth, outputRefs.truth)
        butlerQC.put(outputs.peaks, outputRefs.peaks)

    def run(self, tract, patch, seed, patch_index, bands):
        """Build a mock catalogue, render it at the patch's depth, and return the per-band coadds and truth tables.

        Parameters
        ----------
        tract : `lsst.skymap.TractInfo`
            Tract the coadd belongs to; its WCS is attached to the exposures.
        patch : `lsst.skymap.PatchInfo`
            Patch the coadd fills; its inner bounding box fixes the coadd's position.
        seed : `int`
            Seed and star/noise index of the catalogue realisation (the tract id).
        patch_index : `int`
            Depth in whole years (the patch id); must be 1 to 10.
        bands : `list` [`str`]
            Bands to render, in output order.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            ``coadds`` (`dict` [`str`, `lsst.afw.image.ExposureF`]), ``truth`` and ``peaks``
            (`astropy.table.Table`).
        """
        if not 1 <= patch_index <= MAX_EPOCH_YEAR:
            raise ValueError(f"patch {patch_index} does not name a depth: use patch 1 to {MAX_EPOCH_YEAR} "
                             f"(patch N is the N-year coadd)")
        epoch = f"{patch_index}y"
        corner = patch.getInnerBBox().getMin()
        cfg = {**IMAGE_CONFIG, **{k: v for k, v in self.config.image_config_overrides.items()},
               "psf_model": self.config.psf_model, "canvas_npix": self.config.canvas_npix, "bands": list(bands)}
        cat_cfg = {**{k: v for k, v in self.config.config_overrides.items()},
                   "seed": seed, "npix": self.config.canvas_npix, "out_bands": list(bands)}

        self.log.info("Building mock catalogue (seed %d) and the %s coadd for tract %d, patch %d",
                      seed, epoch, tract.getId(), patch_index)
        cosmos = load_cosmos2025_catalogue(self.config.cosmos_path or DEFAULT_COSMOS_PATH)
        # The pipeline captures the generator's output; the model's own progress bars and prints are noise here.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            catalogue = build_mock_catalogue(cosmos, cat_cfg)
            key, coadd, out_bands, origin, base_image, signal, variance, mask, psf = render_single_coadd(
                catalogue, seed, cfg=cfg, n_workers=self.config.n_workers, epoch=epoch)

        mock = MockCoadd(name=str(tract.getId()), key=key, directory=None, bands=out_bands,
                         origin=(corner.getX(), corner.getY()), pixel_scale=cfg["pixscale"], signal=signal,
                         variance=variance, mask=mask, psf=psf,
                         mask_planes={plane: 1 << bit for plane, bit in MASK_PLANES.items()}, settings=coadd)
        coadds = {band: make_exposure(mock, band, tract.getWcs()) for band in out_bands}

        truth = truth_table(catalogue, out_bands, cfg["pixscale"])
        flagged, found = find_truth_peaks(truth, base_image, origin, coadd, cfg, self.config.min_snr,
                                          self.config.match_radius)
        # Shift the pixel positions into the exposure's frame (origin at the patch corner), so truth x_pix / y_pix
        # line up with the coadd pixels and with detected peak positions; x_array / y_array stay 0-based into the
        # image array.
        for table in (flagged, found):
            table["x_pix"] += corner.getX()
            table["y_pix"] += corner.getY()
        return pipeBase.Struct(coadds=coadds, truth=to_astropy(flagged), peaks=to_astropy(found))
