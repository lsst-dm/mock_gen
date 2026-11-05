"""Tests of the mock catalogue and coadd generator PipelineTasks and their butler repository, on a small mock built
in-process.

The whole module is skipped if the LSST Science Pipelines are missing (including MultiResolutionDetectionTask from
meas_algorithms), so a skip means a needed package is not set up.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    import lsst.afw.image as afwImage
    import lsst.afw.table as afwTable
    import lsst.geom as geom
    import lsst.utils.tests
    from lsst.ctrl.mpexec import SimplePipelineExecutor
    from lsst.daf.butler import Butler
    from lsst.meas.algorithms import DynamicDetectionTask, MultiResolutionDetectionTask
    from lsst.pipe.base import Pipeline

    from lsst.mock.gen.butler import (CATALOGUE_DATASET_TYPES, COADD_DATASET_TYPE, PEAKS_DATASET_TYPE,
                                      TRUTH_DATASET_TYPE, make_repo)
except ImportError as error:
    raise unittest.SkipTest(f"needs the LSST Science Pipelines: {error}")

NPIX = 200
SKYMAP = "mock"
TRACT = 7
OTHER_TRACT = 8
PATCH = 10  # noise 0, the 10-year coadd
NOISE_PATCH = 20  # noise 2, also the 10-year coadd
TRACTS = [TRACT, OTHER_TRACT]
PATCHES = [PATCH, NOISE_PATCH]
OUTPUT = "mock/test"
BANDS = list("ugrizy")
# Fraction of the noise-free r-band peaks with S/N > 10 that a detector must find within 2 pixels.
MIN_RECOVERED = 0.75

_PIPELINE = Path(__file__).parents[1] / "pipelines" / "mock_coadd_gen.yaml"


def recovered_fraction(sources, peaks, band):
    """Fraction of a band's bright truth peaks within 2 px of a detected peak."""
    bright = peaks[(peaks["band"] == band) & (peaks["snr"] > 10)]
    detected = np.array([[peak.getFx(), peak.getFy()]
                         for source in sources for peak in source.getFootprint().getPeaks()])
    distance, _ = cKDTree(detected).query(np.c_[bright["x_pix"], bright["y_pix"]])
    return np.mean(distance < 2)


def run_pipeline(repo, where, overrides=()):
    """Run the two-task pipeline over a repo and data query, returning a butler on the output run."""
    pipeline = Pipeline.fromFile(str(_PIPELINE))
    pipeline.addConfigOverride("mockCatalogGen", "catalogue.npix", NPIX)
    pipeline.addConfigOverride("mockCoaddGen", "image.canvas_npix", NPIX)
    pipeline.addConfigOverride("mockCoaddGen", "n_workers", 1)
    for label, key, value in overrides:
        pipeline.addConfigOverride(label, key, value)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        butler = SimplePipelineExecutor.prep_butler(str(repo), inputs=["skymaps"], output=OUTPUT)
        executor = SimplePipelineExecutor.from_pipeline(pipeline, where=where, butler=butler)
        n_quanta = len(list(executor.quantum_graph))
        executor.run(register_dataset_types=True)
    return Butler.from_config(str(repo), collections=OUTPUT), n_quanta


class MockGenTestCase(lsst.utils.tests.TestCase):
    """Run the generator pipeline on a tiny repo and check what the butler returns."""

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repo = Path(cls.tempdir.name) / "repo"
        make_repo(cls.repo, tracts=TRACTS, patches=PATCHES, npix=NPIX)
        where = f"skymap='{SKYMAP}' AND tract IN ({TRACT},{OTHER_TRACT}) AND patch IN ({PATCH},{NOISE_PATCH})"
        cls.butler, cls.n_quanta = run_pipeline(cls.repo, where)
        cls.dataId = dict(skymap=SKYMAP, tract=TRACT, patch=PATCH)

    @classmethod
    def tearDownClass(cls):
        del cls.butler
        cls.tempdir.cleanup()

    def getExposures(self, dataId=None):
        dataId = dataId or self.dataId
        return {band: self.butler.get(COADD_DATASET_TYPE, dataId, band=band) for band in BANDS}

    def testQuantumGraph(self):
        # 2 tracts x 1 catalogue + 2 tracts x 2 patches of coadds.
        self.assertEqual(self.n_quanta, 2 + 4)

    def testSkyMap(self):
        # make_repo stores a MockSkyMap whose tracts are the (arbitrary) seed ids, so coadd tasks that need a
        # skymap (e.g. the legacy MergeDetectionsTask) can look up a tract by id.
        skyMap = self.butler.get("skyMap", skymap=SKYMAP, collections="skymaps")
        self.assertEqual(skyMap[TRACT].getId(), TRACT)
        bbox = skyMap[TRACT].getPatchInfo(0).getOuterBBox()
        self.assertEqual(bbox.getMin(), geom.Point2I(0, 0))
        self.assertEqual(bbox.getDimensions(), geom.Extent2I(NPIX, NPIX))

    def testCatalogueDatasets(self):
        for name in CATALOGUE_DATASET_TYPES:
            refs = list(self.butler.registry.queryDatasets(name, collections=OUTPUT))
            self.assertEqual({ref.dataId["tract"] for ref in refs}, set(TRACTS))
        galaxies = self.butler.get("mock_galaxies", skymap=SKYMAP, tract=TRACT)
        self.assertIn("type", galaxies.colnames)
        self.assertIn("star", set(galaxies["type"]))

    def testIngestedDatasets(self):
        coadds = list(self.butler.registry.queryDatasets(COADD_DATASET_TYPE, collections=OUTPUT))
        self.assertEqual({ref.dataId["band"] for ref in coadds}, set(BANDS))
        self.assertEqual(len(coadds), len(BANDS) * len(TRACTS) * len(PATCHES))
        for datasetType in (TRUTH_DATASET_TYPE, PEAKS_DATASET_TYPE):
            refs = list(self.butler.registry.queryDatasets(datasetType, collections=OUTPUT))
            self.assertEqual(len(refs), len(TRACTS) * len(PATCHES))

    def testConfigIsRecorded(self):
        for label in ("mockCatalogGen", "mockCoaddGen"):
            configs = list(self.butler.registry.queryDatasets(f"{label}_config", collections=OUTPUT))
            self.assertEqual(len(configs), 1)

    def testExposures(self):
        bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(NPIX, NPIX))
        for band, exposure in self.getExposures().items():
            with self.subTest(band=band):
                self.assertIsInstance(exposure, afwImage.ExposureF)
                self.assertEqual(exposure.getBBox(), bbox)
                self.assertTrue(np.isfinite(exposure.image.array).all())
                self.assertEqual(exposure.getFilter().bandLabel, band)
                self.assertEqual(exposure.getPhotoCalib().getCalibrationMean(), 1.0)
                self.assertAlmostEqual(exposure.getWcs().getPixelScale().asArcseconds(), 0.2, places=6)
                self.assertEqual(exposure.getMetadata()["MOCK_COADD"], "10y_moffat")
                psf = exposure.getPsf()
                kernel = psf.computeKernelImage(psf.getAveragePosition())
                self.assertAlmostEqual(kernel.array.sum(), 1.0, places=6)
                self.assertEqual(np.unravel_index(kernel.array.argmax(), kernel.array.shape),
                                 (kernel.getHeight() // 2, kernel.getWidth() // 2))
        self.assertGreater(float(self.getExposures()["r"].image.array.sum()), 0.0)

    def testTruthTables(self):
        truth = self.butler.get(TRUTH_DATASET_TYPE, self.dataId)
        peaks = self.butler.get(PEAKS_DATASET_TYPE, self.dataId)
        self.assertTrue(set(truth["kind"]) >= {"galaxy", "star"})
        self.assertGreater(len(peaks), 0)
        self.assertIn("snr", peaks.colnames)

    def testNoiseDigitIsIndependent(self):
        # Two patches at the same depth but different noise digit: same catalogue (so same truth), different noise.
        truth = self.butler.get(TRUTH_DATASET_TYPE, self.dataId)
        other = self.butler.get(TRUTH_DATASET_TYPE, skymap=SKYMAP, tract=TRACT, patch=NOISE_PATCH)
        self.assertEqual(len(truth), len(other))
        r0 = self.getExposures()["r"].image.array
        r2 = self.getExposures(dict(skymap=SKYMAP, tract=TRACT, patch=NOISE_PATCH))["r"].image.array
        self.assertFalse(np.allclose(r0, r2))

    def testTractIsIndependent(self):
        r_a = self.getExposures()["r"].image.array
        r_b = self.getExposures(dict(skymap=SKYMAP, tract=OTHER_TRACT, patch=PATCH))["r"].image.array
        self.assertFalse(np.allclose(r_a, r_b))

    def testDynamicDetection(self):
        exposure = self.getExposures()["r"]
        schema = afwTable.SourceTable.makeMinimalSchema()
        task = DynamicDetectionTask(schema=schema)
        result = task.run(afwTable.SourceTable.make(schema), exposure, expId=0)
        peaks = self.butler.get(PEAKS_DATASET_TYPE, self.dataId)
        self.assertGreater(recovered_fraction(result.sources, peaks, "r"), MIN_RECOVERED)

    def testMultiResolutionDetection(self):
        exposures = self.getExposures()
        exposure = afwImage.MultibandExposure.fromExposures(list(exposures), list(exposures.values()))
        schema = afwTable.SourceTable.makeMinimalSchema()
        task = MultiResolutionDetectionTask(schema=schema)
        result = task.run(afwTable.SourceTable.make(schema), exposure)
        peaks = self.butler.get(PEAKS_DATASET_TYPE, self.dataId)
        self.assertGreater(recovered_fraction(result.sources, peaks, "r"), MIN_RECOVERED)


class ConfigEffectTestCase(lsst.utils.tests.TestCase):
    """A first-class config field changes the catalogue: stars off means no star rows."""

    def testStarsOff(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            make_repo(repo, tracts=[TRACT], patches=[PATCH], npix=NPIX)
            where = f"skymap='{SKYMAP}' AND tract={TRACT} AND patch={PATCH}"
            butler, _ = run_pipeline(repo, where, overrides=[("mockCatalogGen", "catalogue.stars", False)])
            galaxies = butler.get("mock_galaxies", skymap=SKYMAP, tract=TRACT)
            self.assertNotIn("star", set(galaxies["type"]))
            truth = butler.get(TRUTH_DATASET_TYPE, skymap=SKYMAP, tract=TRACT, patch=PATCH)
            self.assertNotIn("star", set(truth["kind"]))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
