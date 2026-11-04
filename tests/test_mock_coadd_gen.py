"""Tests of the mock coadd generator PipelineTask and its butler repository, on a small mock built in-process.

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

    from lsst.mock.gen.butler import COADD_DATASET_TYPE, PEAKS_DATASET_TYPE, TRUTH_DATASET_TYPE, make_repo
except ImportError as error:
    raise unittest.SkipTest(f"needs the LSST Science Pipelines: {error}")

NPIX = 200
SKYMAP = "mock"
TRACT = 0
PATCH = 10  # the 10-year coadd
OUTPUT = "mock/test"
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


class MockCoaddGenTestCase(lsst.utils.tests.TestCase):
    """Run the generator pipeline on a tiny repo and check what the butler returns."""

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repo = Path(cls.tempdir.name) / "repo"
        make_repo(cls.repo, n_tracts=1, max_epoch_year=PATCH, npix=NPIX)
        pipeline = Pipeline.fromFile(str(_PIPELINE))
        pipeline.addConfigOverride("mockCoaddGen", "canvas_npix", NPIX)
        pipeline.addConfigOverride("mockCoaddGen", "n_workers", 1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            butler = SimplePipelineExecutor.prep_butler(str(cls.repo), inputs=["skymaps"], output=OUTPUT)
            executor = SimplePipelineExecutor.from_pipeline(
                pipeline, where=f"skymap='{SKYMAP}' AND tract={TRACT} AND patch={PATCH}", butler=butler)
            cls.n_quanta = len(list(executor.quantum_graph))
            executor.run(register_dataset_types=True)
        cls.butler = Butler.from_config(str(cls.repo), collections=OUTPUT)
        cls.dataId = dict(skymap=SKYMAP, tract=TRACT, patch=PATCH)
        cls.bands = list("ugrizy")
        # The patch corner the coadd should sit at.
        skyMap = cls.butler.get("skyMap", skymap=SKYMAP, collections="skymaps")
        cls.corner = skyMap[TRACT][PATCH].getInnerBBox().getMin()

    @classmethod
    def tearDownClass(cls):
        del cls.butler
        cls.tempdir.cleanup()

    def getExposures(self):
        return {band: self.butler.get(COADD_DATASET_TYPE, self.dataId, band=band) for band in self.bands}

    def testNoInputGraphIsOneQuantum(self):
        self.assertEqual(self.n_quanta, 1)

    def testIngestedDatasets(self):
        coadds = list(self.butler.registry.queryDatasets(COADD_DATASET_TYPE, collections=OUTPUT))
        self.assertEqual({ref.dataId["band"] for ref in coadds}, set(self.bands))
        self.assertEqual(len(coadds), len(self.bands))
        for datasetType in (TRUTH_DATASET_TYPE, PEAKS_DATASET_TYPE):
            self.assertEqual(len(list(self.butler.registry.queryDatasets(datasetType, collections=OUTPUT))), 1)

    def testConfigIsRecorded(self):
        # The quantum graph stores each quantum's config, so the repo records how the data was generated.
        configs = list(self.butler.registry.queryDatasets("mockCoaddGen_config", collections=OUTPUT))
        self.assertEqual(len(configs), 1)

    def testExposures(self):
        bbox = geom.Box2I(geom.Point2I(self.corner), geom.Extent2I(NPIX, NPIX))
        for band, exposure in self.getExposures().items():
            with self.subTest(band=band):
                self.assertIsInstance(exposure, afwImage.ExposureF)
                self.assertEqual(exposure.getBBox(), bbox)
                self.assertTrue(np.isfinite(exposure.image.array).all())
                self.assertEqual(exposure.getFilter().bandLabel, band)
                self.assertEqual(exposure.getPhotoCalib().getCalibrationMean(), 1.0)
                self.assertAlmostEqual(exposure.getWcs().getPixelScale().asArcseconds(), 0.2, places=6)
                self.assertEqual(exposure.getMetadata()["MOCK_COADD"], f"{PATCH}y_moffat")
                psf = exposure.getPsf()
                kernel = psf.computeKernelImage(psf.getAveragePosition())
                self.assertAlmostEqual(kernel.array.sum(), 1.0, places=6)
                self.assertEqual(np.unravel_index(kernel.array.argmax(), kernel.array.shape),
                                 (kernel.getHeight()//2, kernel.getWidth()//2))
        self.assertGreater(float(self.getExposures()["r"].image.array.sum()), 0.0)

    def testTruthTables(self):
        truth = self.butler.get(TRUTH_DATASET_TYPE, self.dataId)
        peaks = self.butler.get(PEAKS_DATASET_TYPE, self.dataId)
        self.assertTrue(set(truth["kind"]) >= {"galaxy", "star"})
        self.assertGreater(len(peaks), 0)
        self.assertIn("snr", peaks.colnames)

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


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
