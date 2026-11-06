"""Tests of the detection validation metrics (`lsst.mock.gen.validation`).

The matching and reduction logic is exercised on small in-memory fixtures with
no butler: a fake `TruthSource` and `DetectionReader` return hand-built tables,
so the expected completeness, purity and over-splitting counts can be checked by
hand. The butler readers are thin and covered by the pipeline tests.
"""

import unittest

import numpy as np
import pandas as pd

try:
    import lsst.utils.tests

    from lsst.mock.gen.validation import (DetectionReader, MockTruthSource, TruthSource, completeness,
                                          epoch_label, evaluate, match_patch, purity, purity_by_source,
                                          summarize_purity)
except ImportError as error:
    raise unittest.SkipTest(f"needs the LSST Science Pipelines: {error}")


class FakeTruth(TruthSource):
    """Truth source backed by literal tables, for one patch."""

    supports_purity = True

    def __init__(self, peaks, real, noise, tract=1, patch=0, significance=None, saturated=None):
        self._peaks = peaks
        self._real = real
        self._noise = noise
        self._id = (tract, patch)
        self._significance = significance
        self._saturated = saturated

    def data_ids(self):
        return [self._id]

    def truth_peaks(self, tract, patch):
        return self._peaks.copy()

    def real_sources(self, tract, patch):
        return self._real.copy()

    def noise_maxima(self, tract, patch):
        return self._noise.copy()

    def significance_image(self, tract, patch):
        return self._significance

    def saturated_mask(self, tract, patch):
        return self._saturated


class FakeReader(DetectionReader):
    """Detection reader backed by a literal table, for one patch."""

    def __init__(self, peaks, method):
        self._peaks = peaks
        self.method = method

    def peaks(self, tract, patch):
        return self._peaks.copy()


class EpochLabelTestCase(lsst.utils.tests.TestCase):
    """The patch id's last digit is the epoch."""

    def testLabels(self):
        self.assertEqual(epoch_label(0), "10yr")
        self.assertEqual(epoch_label(10), "10yr")
        self.assertEqual(epoch_label(1), "1yr")
        self.assertEqual(epoch_label(25), "5yr")


class MatchPatchTestCase(lsst.utils.tests.TestCase):
    """Matching a known peak layout gives the expected per-source outcomes."""

    def setUp(self):
        # Two galaxies (groups 100, 101) each with one r-band peak, plus a
        # star-forming region (group 100) near galaxy 100. Galaxy 101's peak
        # has no detection within the radius.
        self.peaks = pd.DataFrame(dict(
            band=["r", "r", "r"],
            x=[10.0, 10.8, 50.0], y=[10.0, 10.0, 50.0],
            snr=[30.0, 12.0, 25.0], truth_id=[0, 1, 2],
            group_key=[100, 100, 101], kind=["galaxy", "sfregion", "galaxy"]))
        self.real = pd.DataFrame(dict(
            x=[10.0, 10.8, 50.0], y=[10.0, 10.0, 50.0],
            group_key=[100, 100, 101], kind=["galaxy", "sfregion", "galaxy"]))
        self.noise = pd.DataFrame(dict(x=[200.0], y=[200.0]))

    def testCompletenessAndBlending(self):
        # One detection on galaxy 100, none on galaxy 101.
        reader = FakeReader(pd.DataFrame(dict(x=[10.1], y=[10.0], significance=[9.0])), "new")
        truth = match_patch(FakeTruth(self.peaks, self.real, self.noise), reader, 1, 0, radius=2.0).truth
        detected = dict(zip(truth["truth_id"], truth["detected"]))
        self.assertTrue(detected[0])   # galaxy 100 peak matched
        self.assertTrue(detected[1])   # its sfregion is within the radius too
        self.assertFalse(detected[2])  # galaxy 101 peak unmatched
        # Galaxy 100 and its sfregion are within the radius of each other.
        blended = dict(zip(truth["truth_id"], truth["blended"]))
        self.assertTrue(blended[0])
        self.assertTrue(blended[1])
        self.assertFalse(blended[2])

    def testGalaxyGranularityFoldsComponents(self):
        reader = FakeReader(pd.DataFrame(dict(x=[10.1], y=[10.0], significance=[9.0])), "new")
        truth = match_patch(FakeTruth(self.peaks, self.real, self.noise), reader, 1, 0, radius=2.0).truth
        edges = np.array([5.0, 20.0, 100.0])
        curve = completeness(truth, edges, granularity="galaxy", min_count=1)
        # Group 100 (two peaks, brightest S/N 30) is recovered; group 101 is not.
        recovered = curve[(curve["x"] > 20) & np.isfinite(curve["completeness"])]
        self.assertEqual(float(recovered["completeness"].iloc[0]), 0.5)

    def testPurityAndFalsePositiveSplit(self):
        # Three detections: on galaxy 100 (real), on the noise maximum, and in
        # empty sky (spurious).
        reader = FakeReader(pd.DataFrame(dict(x=[10.0, 200.0, 500.0], y=[10.0, 200.0, 500.0])), "new")
        det = match_patch(FakeTruth(self.peaks, self.real, self.noise), reader, 1, 0, radius=2.0).detections
        self.assertEqual(det["matched"].tolist(), [True, False, False])
        self.assertEqual(det["noise_maximum"].tolist(), [False, True, False])
        summary = summarize_purity(det)
        self.assertAlmostEqual(float(summary["purity"].iloc[0]), 1 / 3)
        self.assertEqual(int(summary["n_noise_maximum"].iloc[0]), 1)
        self.assertEqual(int(summary["n_spurious"].iloc[0]), 1)

    def testExcludeSaturated(self):
        # Galaxy 101 (truth_id 2) sits on a saturated pixel; excluding it drops
        # that peak from the truth table so it is not counted as a miss.
        saturated = np.zeros((64, 64), dtype=bool)
        saturated[50, 50] = True
        reader = FakeReader(pd.DataFrame(dict(x=[10.0], y=[10.0])), "new")
        truth = FakeTruth(self.peaks, self.real, self.noise, saturated=saturated)
        kept = match_patch(truth, reader, 1, 0, radius=2.0, exclude_saturated=True).truth
        self.assertNotIn(2, kept["truth_id"].tolist())
        self.assertEqual(sorted(kept["truth_id"].tolist()), [0, 1])


class EvaluateTestCase(lsst.utils.tests.TestCase):
    """`evaluate` concatenates methods and skips a method missing a patch."""

    def testMissingPatchIsSkipped(self):
        peaks = pd.DataFrame(dict(band=["r"], x=[10.0], y=[10.0], snr=[30.0], truth_id=[0],
                                  group_key=[100], kind=["galaxy"]))
        real = pd.DataFrame(dict(x=[10.0], y=[10.0], group_key=[100], kind=["galaxy"]))
        noise = pd.DataFrame(dict(x=[200.0], y=[200.0]))
        truth = FakeTruth(peaks, real, noise)

        class MissingReader(DetectionReader):
            def __init__(self):
                self.method = "legacy"

            def peaks(self, tract, patch):
                raise LookupError("no dataset")

        found = FakeReader(pd.DataFrame(dict(x=[10.0], y=[10.0])), "new")
        matches = evaluate(truth, [found, MissingReader()])
        self.assertEqual(sorted(matches.truth["method"].unique()), ["new"])

    def testPurityCurveBins(self):
        peaks = pd.DataFrame(dict(band=["r"], x=[10.0], y=[10.0], snr=[30.0], truth_id=[0],
                                  group_key=[100], kind=["galaxy"]))
        real = pd.DataFrame(dict(x=[10.0], y=[10.0], group_key=[100], kind=["galaxy"]))
        noise = pd.DataFrame(dict(x=[200.0], y=[200.0]))
        # Significance comes from the shared image: one real detection at (10, 10)
        # and one spurious at (500, 500), both at significance 9.
        significance = np.zeros((512, 512))
        significance[10, 10] = significance[500, 500] = 9.0
        reader = FakeReader(pd.DataFrame(dict(x=[10.0, 500.0], y=[10.0, 500.0])), "new")
        det = match_patch(FakeTruth(peaks, real, noise, significance=significance),
                          reader, 1, 0, radius=2.0).detections
        curve = purity(det, np.array([5.0, 15.0]), min_count=1)
        self.assertAlmostEqual(float(curve["purity"].iloc[0]), 0.5)


class PurityBySourceTestCase(lsst.utils.tests.TestCase):
    """`purity_by_source` splits the new detection's purity by scale and clique."""

    def _detections(self, scale, structure_peaks, matched):
        return pd.DataFrame(dict(method="new", epoch="10yr", scale=scale,
                                 structure_peaks=structure_peaks, matched=matched))

    def testByScale(self):
        det = self._detections(scale=[1, 1, 2, 2], structure_peaks=[1, 1, 1, 1],
                               matched=[True, True, True, False])
        split = purity_by_source(det, "scale").set_index("scale")
        self.assertAlmostEqual(float(split.loc[1, "purity"]), 1.0)
        self.assertAlmostEqual(float(split.loc[2, "purity"]), 0.5)

    def testByStructureAndLegacyDropped(self):
        det = self._detections(scale=[1, 1, 1], structure_peaks=[1, 2, 2],
                               matched=[True, True, False])
        legacy = pd.DataFrame(dict(method="legacy", epoch="10yr", scale=[-1], structure_peaks=[-1],
                                   matched=[False]))
        split = purity_by_source(pd.concat([det, legacy], ignore_index=True),
                                 "structure_peaks").set_index("split")
        self.assertAlmostEqual(float(split.loc["1", "purity"]), 1.0)
        self.assertAlmostEqual(float(split.loc[">1", "purity"]), 0.5)
        # Legacy (scale -1) is excluded, so only the two new buckets appear.
        self.assertEqual(int(split["n"].sum()), 3)


class RealSourceModeTestCase(lsst.utils.tests.TestCase):
    """`MockTruthSource` real_source_mode selects which components count as real."""

    class _StubButler:
        def __init__(self, truth):
            self._truth = truth

        def get(self, dataset, **kwargs):
            class _Wrap:
                def __init__(self, df):
                    self._df = df

                def to_pandas(self):
                    return self._df
            return _Wrap(self._truth)

    def _source(self, mode):
        # Two galaxies: one with a noise-free peak, one without (a pure body).
        truth = pd.DataFrame(dict(kind=["galaxy", "galaxy"], catalogue_id=[1, 2], parent_id=[-1, -1],
                                  x_pix=[10.0, 20.0], y_pix=[10.0, 20.0], n_peak_bands=[2, 0]))
        return MockTruthSource(self._StubButler(truth), "c", real_source_mode=mode)

    def testPeaksModeKeepsOnlyPeaked(self):
        real = self._source("peaks").real_sources(1, 0)
        self.assertEqual(real["x"].tolist(), [10.0])

    def testComponentsModeKeepsAll(self):
        real = self._source("components").real_sources(1, 0)
        self.assertEqual(sorted(real["x"].tolist()), [10.0, 20.0])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
