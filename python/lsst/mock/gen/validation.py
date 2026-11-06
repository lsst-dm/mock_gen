"""Detection completeness and purity against truth, for new vs legacy detection.

This compares a detection algorithm's peaks against a truth catalogue by
matching peaks to truth positions within a pixel radius, the same way for every
method so the results are comparable. It serves two truth sources behind one
interface (`TruthSource`): the fully simulated mocks of this package
(`MockTruthSource`, reading ``mock_truth`` / ``mock_truth_peaks``) and real
coadds with injected sources (`InjectedTruthSource`). The mock case alone can
measure purity and tell a galaxy's own peak from the peaks of its star-forming
regions and tidal blobs, because only there is every component and every
noise-free local maximum known.

The matching core (`match_patch`) runs per (tract, patch) and returns two tidy
tables, one truth-side and one detection-side (`PatchMatches`); every metric
(completeness, purity, over-splitting, positional accuracy, blending) is a
group-and-reduce over the concatenation of those tables across patches
(`evaluate`). Coordinates are matched within a patch only: a mock tract and
patch share one frame at the image origin, so peaks from different patches would
collide if pooled.
"""

__all__ = [
    "DetectionReader",
    "InjectedTruthSource",
    "LegacyDetectionReader",
    "MockTruthSource",
    "NewDetectionReader",
    "PatchMatches",
    "TruthSource",
    "completeness",
    "epoch_label",
    "evaluate",
    "match_distance_histogram",
    "match_patch",
    "plot_completeness",
    "purity",
    "purity_by_source",
    "split_examples",
    "summarize_purity",
]

from dataclasses import dataclass

import numpy as np
import pandas as pd
from mock_lsst_image_generation.photometry import flux_njy_to_mag
from scipy.spatial import cKDTree

import lsst.geom as geom
from lsst.scarlet.lite.detect import build_chi2_significance

# Match radius in pixels: a truth peak and a detection peak are the same source
# if they are within this. Set by the recovered_fraction check in
# tests/test_mock_coadd_gen.py; the PSF core is a few pixels across.
DEFAULT_RADIUS = 2.0
BANDS = tuple("ugrizy")

# Epoch gets a fixed colourblind-safe colour (Wong palette) and method gets a
# line style, so a curve's identity never depends on colour alone and the same
# epoch keeps its colour across panels.
_EPOCH_COLOR = {"1yr": "#E69F00", "5yr": "#0072B2", "10yr": "#009E73"}
_METHOD_STYLE = {"new": "-", "legacy": "--"}
_METHOD_MARKER = {"new": "o", "legacy": "s"}


def epoch_label(patch):
    """Survey epoch a patch id encodes.

    Parameters
    ----------
    patch : `int`
        Patch id; its last decimal digit is the epoch (0 is the 10 year coadd),
        the rest is the noise realisation.

    Returns
    -------
    label : `str`
        Epoch as ``"10yr"``, ``"1yr"`` and so on.
    """
    digit = patch % 10
    return "10yr" if digit == 0 else f"{digit}yr"


@dataclass
class PatchMatches:
    """Truth-side and detection-side match tables for one patch and method.

    Parameters
    ----------
    truth : `pandas.DataFrame`
        One row per (truth peak, band): ``tract``, ``patch``, ``epoch``,
        ``method``, ``band``, ``truth_id``, ``group_key``, ``kind``, ``snr``,
        ``mag``, ``x``, ``y``, ``detected`` (a detection within the radius),
        ``match_dist`` and ``blended`` (another truth peak of the same band
        within the radius).
    detections : `pandas.DataFrame`
        One row per detection: ``tract``, ``patch``, ``epoch``, ``method``,
        ``x``, ``y``, ``significance`` (the chi significance image value at the
        peak, common to both methods), ``matched`` (near a real source),
        ``group_key``, ``kind``, ``match_dist`` and ``noise_maximum`` (an
        unmatched detection that instead lands on a known noise-free maximum).
    """

    truth: pd.DataFrame
    detections: pd.DataFrame


class TruthSource:
    """Truth peaks, real sources and noise maxima for a detection comparison.

    A subclass adapts one kind of truth (simulated or injected) to the three
    tables `match_patch` needs. Positions are in the coadd pixel frame.
    """

    #: Whether this source can measure purity (knows the full real-source list).
    supports_purity = False

    def data_ids(self):
        """Patches with truth, as ``(tract, patch)`` pairs."""
        raise NotImplementedError

    def truth_peaks(self, tract, patch):
        """Per-band truth peaks to match detections against for completeness.

        Parameters
        ----------
        tract, patch : `int`
            Data id.

        Returns
        -------
        peaks : `pandas.DataFrame`
            Columns ``band``, ``x``, ``y``, ``snr``, ``mag`` (the component's
            AB magnitude in the peak's band), ``truth_id``, ``group_key`` (the
            source a peak is grouped under; see `MockTruthSource`) and ``kind``.
        """
        raise NotImplementedError

    def real_sources(self, tract, patch):
        """Real-source positions to match detections against for purity.

        Returns `None` when the source cannot enumerate real sources, in which
        case purity is skipped.

        Parameters
        ----------
        tract, patch : `int`
            Data id.

        Returns
        -------
        sources : `pandas.DataFrame` or `None`
            Columns ``x``, ``y``, ``group_key``, ``kind``.
        """
        return None

    def noise_maxima(self, tract, patch):
        """Noise-free local maxima that match no real source.

        These separate a genuine false positive from a detection that landed on
        real but uncatalogued faint structure. Returns `None` when unknown.

        Parameters
        ----------
        tract, patch : `int`
            Data id.

        Returns
        -------
        maxima : `pandas.DataFrame` or `None`
            Columns ``x``, ``y``.
        """
        return None

    def significance_image(self, tract, patch):
        """Chi detection-significance image both methods' peaks are scored on.

        One image in units of Gaussian sigma, so a peak from either method is
        read on the same scale. Returns `None` when the coadds are unavailable,
        in which case no significance is attached to detections.

        Parameters
        ----------
        tract, patch : `int`
            Data id.

        Returns
        -------
        image : `numpy.ndarray` or `None`
            ``(Ny, Nx)`` significance, indexed ``[y, x]``.
        """
        return None

    def saturated_mask(self, tract, patch):
        """Pixels where the coadd is saturated, as a boolean image.

        Detection excludes saturated pixels (the new task also dilates them),
        so a truth source whose peak sits on one cannot be detected; completeness
        can drop it with ``exclude_saturated``. Returns `None` when unknown.

        Parameters
        ----------
        tract, patch : `int`
            Data id.

        Returns
        -------
        mask : `numpy.ndarray` or `None`
            ``(Ny, Nx)`` boolean, `True` where saturated, indexed ``[y, x]``.
        """
        return None


class MockTruthSource(TruthSource):
    """Truth from the simulated mocks (``mock_truth`` and ``mock_truth_peaks``).

    A galaxy's star-forming regions and tidal blobs are grouped under the
    galaxy so a method is not charged for failing to split one galaxy into its
    pieces: ``group_key`` is the host galaxy (or star) for every component. The
    peak table already carries the per-band S/N of each noise-free maximum and
    the component it belongs to (``truth_id``, -1 for a maximum with no
    component), so completeness, purity and the noise-maximum cross-reference
    all come from the butler with no image handling.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to read from.
    collection : `str`
        Collection holding ``mock_truth`` and ``mock_truth_peaks``.
    skymap : `str`, optional
        Skymap name.
    real_source_mode : `str`, optional
        Which truth positions count as a real source for purity. ``"peaks"``
        (default) keeps only components with a noise-free maximum in some band,
        so a detection is pure only if it recovered a peak. ``"components"``
        keeps every component, so a detection on real galaxy or star-forming
        region light that is offset from the catalogued peak still counts;
        comparing the two separates genuine false positives from peaks on real
        structure the match radius missed.
    """

    supports_purity = True

    def __init__(self, butler, collection, skymap="mock", real_source_mode="peaks"):
        self._butler = butler
        self._collection = collection
        self._skymap = skymap
        self._real_source_mode = real_source_mode
        self._image_cache = {}

    def data_ids(self):
        refs = self._butler.registry.queryDatasets(
            "mock_truth_peaks", collections=self._collection, findFirst=True)
        return sorted({(ref.dataId["tract"], ref.dataId["patch"]) for ref in refs})

    def _truth(self, tract, patch):
        return self._butler.get("mock_truth", skymap=self._skymap, tract=tract, patch=patch,
                                collections=self._collection).to_pandas()

    @staticmethod
    def _group_key(truth):
        # A galaxy or star is its own group (catalogue_id); a star-forming region
        # or tidal blob groups under its host galaxy (parent_id). The two id
        # spaces do not overlap, so one integer column keys every component.
        is_parent = truth["kind"].isin(["galaxy", "star"])
        return np.where(is_parent, truth["catalogue_id"], truth["parent_id"]).astype(int)

    def truth_peaks(self, tract, patch):
        truth = self._truth(tract, patch)
        truth["group_key"] = self._group_key(truth)
        flux_columns = [f"flux_{band}" for band in BANDS if f"flux_{band}" in truth.columns]
        peaks = self._butler.get("mock_truth_peaks", skymap=self._skymap, tract=tract, patch=patch,
                                 collections=self._collection).to_pandas()
        peaks = peaks[peaks["truth_id"] >= 0]
        keyed = peaks.merge(truth[["truth_id", "group_key", "kind"] + flux_columns], on="truth_id", how="left")
        # The component's magnitude in the peak's own band (bulge + disc only for
        # a galaxy; the region's or star's own flux otherwise).
        flux = keyed.apply(lambda row: row[f"flux_{row['band']}"], axis=1).to_numpy(float)
        return pd.DataFrame(dict(band=keyed["band"], x=keyed["x_pix"], y=keyed["y_pix"],
                                 snr=keyed["snr"], mag=flux_njy_to_mag(flux), truth_id=keyed["truth_id"],
                                 group_key=keyed["group_key"], kind=keyed["kind"]))

    def real_sources(self, tract, patch):
        truth = self._truth(tract, patch)
        if self._real_source_mode == "components":
            # Every component counts, so a detection on real galaxy or
            # star-forming-region light offset from the catalogued peak is pure.
            real = truth
        else:
            # A component with no noise-free maximum in any band is not a local
            # peak, so a detection on it is not a recovery of a real source.
            real = truth[truth["n_peak_bands"] > 0]
        return pd.DataFrame(dict(x=real["x_pix"], y=real["y_pix"],
                                 group_key=self._group_key(real), kind=real["kind"]))

    def noise_maxima(self, tract, patch):
        peaks = self._butler.get("mock_truth_peaks", skymap=self._skymap, tract=tract, patch=patch,
                                 collections=self._collection).to_pandas()
        noise = peaks[peaks["truth_id"] < 0]
        return pd.DataFrame(dict(x=noise["x_pix"], y=noise["y_pix"]))

    def _images(self, tract, patch):
        # Read the six coadds once per patch and derive both the chi
        # significance and the saturated mask; evaluate scores both methods on
        # the same patch, so without this each is read twice.
        if (tract, patch) not in self._image_cache:
            coadds = [self._butler.get("deepCoadd", skymap=self._skymap, tract=tract, patch=patch,
                                       band=band, collections=self._collection) for band in BANDS]
            standardized = np.array([c.image.array / np.sqrt(c.variance.array) for c in coadds])
            # The chi coadd scarlet lite detects on: standardize each band by its
            # pixel noise and clip-and-sum to Gaussian sigma. Lower than the new
            # catalogue's own peak_sigma (dominated by the brightest band), but
            # one scale both methods share.
            significance = build_chi2_significance(standardized)
            sat_bit = coadds[0].mask.getPlaneBitMask("SAT")
            saturated = np.logical_or.reduce([(c.mask.array & sat_bit) > 0 for c in coadds])
            self._image_cache[(tract, patch)] = (significance, saturated)
        return self._image_cache[(tract, patch)]

    def significance_image(self, tract, patch):
        return self._images(tract, patch)[0]

    def saturated_mask(self, tract, patch):
        return self._images(tract, patch)[1]


class InjectedTruthSource(TruthSource):
    """Truth from injected sources on real coadds.

    A port of the injected-truth reader in the DM-52721 analysis scripts: the
    injected catalogue carries sky positions, mapped to the coadd pixel frame
    through the coadd WCS, and a reference magnitude in place of a per-band S/N.
    Injected sources have no sub-component structure and no noise-free maximum
    list, so every source is its own group and purity is not measured.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to read from.
    collection : `str`
        Collection holding the injected catalogue and coadds.
    catalog_dataset : `str`, optional
        Injected catalogue dataset type (tract dimensions).
    coadd_dataset : `str`, optional
        Coadd dataset type whose WCS maps sky to pixels.
    ref_band : `str`, optional
        Band whose ``{band}_mag`` is the magnitude axis; injected truth has no
        per-band noise-free S/N, so ``snr`` is NaN.
    skymap : `str`, optional
        Skymap name.
    """

    def __init__(self, butler, collection, catalog_dataset="injected_deep_coadd_predetection_catalog_tract",
                 coadd_dataset="injected_deep_coadd", ref_band="i", skymap="lsst_cells_v2"):
        self._butler = butler
        self._collection = collection
        self._catalog_dataset = catalog_dataset
        self._coadd_dataset = coadd_dataset
        self._ref_band = ref_band
        self._skymap = skymap
        self._wcs_cache = {}

    def data_ids(self):
        refs = self._butler.registry.queryDatasets(
            self._coadd_dataset, collections=self._collection, findFirst=True)
        return sorted({(ref.dataId["tract"], ref.dataId["patch"]) for ref in refs})

    def _wcs(self, tract, patch):
        if (tract, patch) not in self._wcs_cache:
            coadd = self._butler.get(self._coadd_dataset, skymap=self._skymap, tract=tract, patch=patch,
                                     band=self._ref_band, collections=self._collection)
            self._wcs_cache[(tract, patch)] = coadd.getWcs()
        return self._wcs_cache[(tract, patch)]

    def truth_peaks(self, tract, patch):
        cat = self._butler.get(self._catalog_dataset, skymap=self._skymap, tract=tract,
                               collections=self._collection).to_pandas()
        cat = cat[cat["injected_isPrimary"]] if "injected_isPrimary" in cat else cat
        wcs = self._wcs(tract, patch)
        xy = np.array([wcs.skyToPixel(geom.SpherePoint(float(ra), float(dec), geom.degrees))
                       for ra, dec in zip(cat["ra"], cat["dec"])])
        mag = cat[f"{self._ref_band}_mag"].to_numpy(float)
        return pd.DataFrame(dict(band=self._ref_band, x=xy[:, 0], y=xy[:, 1],
                                 snr=np.full(len(cat), np.nan), mag=mag,
                                 truth_id=np.arange(len(cat)), group_key=np.arange(len(cat)),
                                 kind="injected"))


class DetectionReader:
    """A detection method's peaks in the coadd pixel frame, per patch.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to read from.
    collection : `str`
        Collection holding the method's output.
    method : `str`
        Label carried through to the result tables (e.g. ``"new"``).
    skymap : `str`, optional
        Skymap name.
    """

    def __init__(self, butler, collection, method, skymap="mock"):
        self._butler = butler
        self._collection = collection
        self.method = method
        self._skymap = skymap

    def peaks(self, tract, patch):
        """Peak positions for one patch.

        Parameters
        ----------
        tract, patch : `int`
            Data id.

        Returns
        -------
        peaks : `pandas.DataFrame`
            Columns ``x`` and ``y`` in the coadd pixel frame. Detection
            significance is not read here: `match_patch` assigns it from one
            chi significance image common to both methods, so the two are
            comparable (each method's own peak statistic is a different
            quantity: new stores a per-source S/N, legacy a peak pixel flux).
        """
        raise NotImplementedError


class NewDetectionReader(DetectionReader):
    """Peaks from the multi-band `MultiBandDetectionTask` merged peak table.

    The peaks carry three diagnostic columns the legacy catalogue has no
    analogue for: ``scale`` (the starlet scale the peak was found on; higher is
    lower frequency), ``structure_peaks`` (the number of peaks in the peak's
    connected structure, so ``> 1`` marks a peak the clique cover split off a
    larger structure) and ``structure_id`` (the structure the peak belongs to).
    The first two bear on purity; ``structure_id`` groups the split peaks of one
    structure for inspection.
    """

    def peaks(self, tract, patch):
        table = self._butler.get("object_detection_merged_peaks", skymap=self._skymap, tract=tract,
                                 patch=patch, collections=self._collection).to_pandas()
        table = table[table["polarity"] == 1]
        return pd.DataFrame(dict(x=table["x"].to_numpy(float), y=table["y"].to_numpy(float),
                                 scale=table["scale"].to_numpy(int),
                                 structure_peaks=table["structure_peaks"].to_numpy(int),
                                 structure_id=table["structure_id"].to_numpy(int)))


class LegacyDetectionReader(DetectionReader):
    """Peaks from the legacy per-band detect and merge (``object_detection_merged_legacy``).

    ``scale``, ``structure_peaks`` and ``structure_id`` are filled with -1: the
    legacy merge has no starlet scale or clique structure, so the diagnostic
    splits that use them apply to the new detection only.
    """

    def peaks(self, tract, patch):
        merged = self._butler.get("object_detection_merged_legacy", skymap=self._skymap, tract=tract,
                                  patch=patch, collections=self._collection)
        xs, ys = [], []
        for source in merged:
            for peak in source.getFootprint().getPeaks():
                xs.append(peak["i_x"])
                ys.append(peak["i_y"])
        return pd.DataFrame(dict(x=np.array(xs, float), y=np.array(ys, float),
                                 scale=np.full(len(xs), -1), structure_peaks=np.full(len(xs), -1),
                                 structure_id=np.full(len(xs), -1)))


def _nearest(query_xy, target_xy, radius):
    """Distance and index of the nearest target within radius, else (inf, -1)."""
    if len(query_xy) == 0:
        return np.zeros(0), np.zeros(0, int)
    if len(target_xy) == 0:
        return np.full(len(query_xy), np.inf), np.full(len(query_xy), -1)
    dist, idx = cKDTree(target_xy).query(query_xy, k=1, distance_upper_bound=radius)
    idx = np.where(np.isfinite(dist), idx, -1)
    return dist, idx


def _significance_at(image, xy):
    """Significance-image value at integer-rounded pixel positions, 0 off frame."""
    if image is None or len(xy) == 0:
        return np.zeros(len(xy))
    ny, nx = image.shape
    x = np.clip(np.round(xy[:, 0]).astype(int), 0, nx - 1)
    y = np.clip(np.round(xy[:, 1]).astype(int), 0, ny - 1)
    return np.asarray(image[y, x], dtype=float)


def match_patch(truth_source, reader, tract, patch, radius=DEFAULT_RADIUS, exclude_saturated=False):
    """Match one method's peaks against truth for one patch.

    Parameters
    ----------
    truth_source : `TruthSource`
        Truth for this comparison.
    reader : `DetectionReader`
        The method whose peaks are matched.
    tract, patch : `int`
        Data id.
    radius : `float`, optional
        Match radius in pixels.
    exclude_saturated : `bool`, optional
        Drop truth peaks whose pixel is saturated. Detection masks saturated
        pixels (the new task dilates them too), so such a source cannot be
        detected by either method and would only depress completeness; the
        methods differ in how far they grow the mask, so leaving these in
        measures the mask, not the detection.

    Returns
    -------
    matches : `PatchMatches`
        The truth-side and detection-side tables for this patch and method.
    """
    epoch = epoch_label(patch)
    truth = truth_source.truth_peaks(tract, patch).reset_index(drop=True)
    det = reader.peaks(tract, patch).reset_index(drop=True)
    det_xy = det[["x", "y"]].to_numpy(float)

    if exclude_saturated:
        saturated = truth_source.saturated_mask(tract, patch)
        if saturated is not None:
            keep = ~_significance_at(saturated.astype(float), truth[["x", "y"]].to_numpy(float)).astype(bool)
            truth = truth[keep].reset_index(drop=True)

    significance_image = truth_source.significance_image(tract, patch)
    det["significance"] = _significance_at(significance_image, det_xy)

    # Completeness: each truth peak to its nearest detection, within its band so
    # a peak is not matched to a detection of a blend it does not belong to.
    detected = np.zeros(len(truth), dtype=bool)
    match_dist = np.full(len(truth), np.nan)
    blended = np.zeros(len(truth), dtype=bool)
    for band in truth["band"].unique():
        rows = np.flatnonzero((truth["band"] == band).to_numpy())
        band_xy = truth.loc[rows, ["x", "y"]].to_numpy(float)
        dist, idx = _nearest(band_xy, det_xy, radius)
        detected[rows] = idx >= 0
        match_dist[rows] = np.where(idx >= 0, dist, np.nan)
        # A peak is blended if another truth peak of the same band is within the
        # radius (the second neighbour, since the first is the peak itself).
        if len(rows) > 1:
            neigh, _ = cKDTree(band_xy).query(band_xy, k=2)
            blended[rows] = neigh[:, 1] <= radius
    truth_table = truth.assign(tract=tract, patch=patch, epoch=epoch, method=reader.method,
                               detected=detected, match_dist=match_dist, blended=blended)

    # Purity: each detection to its nearest real source, over all bands at once.
    real = truth_source.real_sources(tract, patch)
    if real is not None:
        real = real.reset_index(drop=True)
        dist, idx = _nearest(det_xy, real[["x", "y"]].to_numpy(float), radius)
        matched = idx >= 0
        group = np.where(matched, real["group_key"].to_numpy()[np.where(matched, idx, 0)], -1)
        kind = np.where(matched, real["kind"].to_numpy()[np.where(matched, idx, 0)], "")
        noise = truth_source.noise_maxima(tract, patch)
        noise_maximum = np.zeros(len(det), dtype=bool)
        if noise is not None and len(noise):
            ndist, nidx = _nearest(det_xy, noise[["x", "y"]].to_numpy(float), radius)
            noise_maximum = (~matched) & (nidx >= 0)
        det_table = det.assign(tract=tract, patch=patch, epoch=epoch, method=reader.method,
                               matched=matched, group_key=group, kind=kind,
                               match_dist=np.where(matched, dist, np.nan), noise_maximum=noise_maximum)
    else:
        det_table = det.assign(tract=tract, patch=patch, epoch=epoch, method=reader.method,
                               matched=pd.NA, group_key=pd.NA, kind=pd.NA, match_dist=np.nan,
                               noise_maximum=pd.NA)
    return PatchMatches(truth_table, det_table)


def evaluate(truth_source, readers, radius=DEFAULT_RADIUS, data_ids=None, exclude_saturated=False):
    """Match every method against truth over every patch.

    Parameters
    ----------
    truth_source : `TruthSource`
        Truth for the comparison.
    readers : `list` [`DetectionReader`]
        The methods to compare.
    radius : `float`, optional
        Match radius in pixels.
    data_ids : `list` [`tuple`], optional
        ``(tract, patch)`` pairs to run; the truth source's own list by default.
    exclude_saturated : `bool`, optional
        Drop truth peaks on saturated pixels (see `match_patch`).

    Returns
    -------
    matches : `PatchMatches`
        The truth-side and detection-side tables concatenated over all patches
        and methods. A method absent for a patch (no dataset) is skipped.
    """
    ids = data_ids if data_ids is not None else truth_source.data_ids()
    truth_parts, det_parts = [], []
    for tract, patch in ids:
        for reader in readers:
            try:
                result = match_patch(truth_source, reader, tract, patch, radius, exclude_saturated)
            except LookupError:
                # The method has no output for this patch; leave its curve to the
                # patches it does cover rather than failing the whole run.
                continue
            truth_parts.append(result.truth)
            det_parts.append(result.detections)
    return PatchMatches(pd.concat(truth_parts, ignore_index=True),
                        pd.concat(det_parts, ignore_index=True))


def _snr_bins(values, edges):
    """Bin centres and index per value for a set of bin edges."""
    idx = np.digitize(values, edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    return idx, centres


def completeness(truth, edges, granularity="peak", x="snr", min_count=10):
    """Completeness per band, epoch and method as a function of brightness.

    Parameters
    ----------
    truth : `pandas.DataFrame`
        The truth-side table from `evaluate`.
    edges : `numpy.ndarray`
        Bin edges, in the units of ``x``.
    granularity : `str`, optional
        ``"peak"`` counts each truth peak; ``"galaxy"`` counts each source
        group (star-forming regions and tidal blobs folded into their galaxy),
        recovered if any of its peaks in that band is detected, binned on the
        group's brightest peak.
    x : `str`, optional
        Brightness axis to bin on: ``"snr"`` (truth peak S/N) or ``"mag"``
        (AB magnitude). Returned as the ``x`` column of the curve.
    min_count : `int`, optional
        Bins with fewer truth entries are returned as NaN, since a bin of one
        or two is 0 or 1 by chance.

    Returns
    -------
    curve : `pandas.DataFrame`
        Columns ``band``, ``epoch``, ``method``, ``x`` (bin centre), ``axis``
        (the name of the binning quantity), ``completeness`` and ``count``.
    """
    df = truth
    if granularity == "galaxy":
        # The brightest peak of a group is the largest S/N but the smallest
        # magnitude, so pick the extreme that means brightest for this axis.
        brightest = "min" if x == "mag" else "max"
        df = (df.groupby(["band", "epoch", "method", "group_key"], as_index=False)
                .agg(**{x: (x, brightest), "detected": ("detected", "any")}))
    rows = []
    for (band, epoch, method), group in df.groupby(["band", "epoch", "method"]):
        idx, centres = _snr_bins(group[x].to_numpy(float), edges)
        for b, centre in enumerate(centres):
            sel = idx == b
            n = int(sel.sum())
            frac = float(group["detected"].to_numpy()[sel].mean()) if n >= min_count else np.nan
            rows.append(dict(band=band, epoch=epoch, method=method, x=centre, axis=x,
                             completeness=frac, count=n))
    return pd.DataFrame(rows)


def purity(detections, edges, min_count=10):
    """Purity per epoch and method as a function of detection significance.

    Significance is the chi significance image value at each peak, so both
    methods are on one axis and their curves are directly comparable.

    Parameters
    ----------
    detections : `pandas.DataFrame`
        The detection-side table from `evaluate`; purity needs a truth source
        that supports it (``matched`` not null).
    edges : `numpy.ndarray`
        Significance bin edges, in Gaussian sigma.
    min_count : `int`, optional
        Bins with fewer detections are returned as NaN.

    Returns
    -------
    curve : `pandas.DataFrame`
        Columns ``epoch``, ``method``, ``significance`` (bin centre),
        ``purity`` and ``count``.
    """
    rows = []
    for (epoch, method), group in detections.groupby(["epoch", "method"]):
        idx, centres = _snr_bins(group["significance"].to_numpy(float), edges)
        matched = group["matched"].to_numpy(float)
        for b, centre in enumerate(centres):
            sel = idx == b
            n = int(sel.sum())
            frac = float(matched[sel].mean()) if n >= min_count else np.nan
            rows.append(dict(epoch=epoch, method=method, significance=centre, purity=frac, count=n))
    return pd.DataFrame(rows)


def summarize_purity(detections):
    """Overall purity and false-positive breakdown per epoch and method.

    Parameters
    ----------
    detections : `pandas.DataFrame`
        The detection-side table from `evaluate`.

    Returns
    -------
    summary : `pandas.DataFrame`
        Columns ``epoch``, ``method``, ``n`` (detections), ``purity``
        (fraction matched to a real source), ``n_false`` (unmatched),
        ``n_noise_maximum`` (unmatched but on a known noise-free maximum) and
        ``n_spurious`` (unmatched and not on any maximum).
    """
    rows = []
    for (epoch, method), group in detections.groupby(["epoch", "method"]):
        matched = group["matched"].to_numpy(bool)
        noise = group["noise_maximum"].to_numpy(bool)
        n = len(group)
        rows.append(dict(epoch=epoch, method=method, n=n,
                         purity=float(matched.mean()) if n else np.nan,
                         n_false=int((~matched).sum()),
                         n_noise_maximum=int(noise.sum()),
                         n_spurious=int(((~matched) & (~noise)).sum())))
    return pd.DataFrame(rows)


def purity_by_source(detections, by):
    """Purity of the new detection split by a diagnostic column.

    Attributes the new detection's false positives to the starlet scale a peak
    was found on (``scale``) or to clique over-splitting (``structure_peaks``).
    Only the new method carries these; legacy peaks (filled -1) are dropped.

    Parameters
    ----------
    detections : `pandas.DataFrame`
        The detection-side table from `evaluate`.
    by : `str`
        Column to split on: ``"scale"``, or ``"structure_peaks"`` (bucketed
        into ``1`` and ``>1``).

    Returns
    -------
    split : `pandas.DataFrame`
        Columns ``epoch``, the split key, ``n`` and ``purity``.
    """
    new = detections[detections["method"] == "new"].copy()
    new = new[new["scale"] >= 0]
    key = by
    if by == "structure_peaks":
        key = "split"
        new[key] = np.where(new["structure_peaks"] > 1, ">1", "1")
    rows = []
    for (epoch, value), group in new.groupby(["epoch", key]):
        rows.append(dict(epoch=epoch, **{key: value}, n=len(group),
                         purity=float(group["matched"].mean())))
    return pd.DataFrame(rows)


def plot_completeness(curve, band, ax=None, title=None):
    """Plot the completeness curves of one band, one line per (epoch, method).

    Parameters
    ----------
    curve : `pandas.DataFrame`
        Output of `completeness`.
    band : `str`
        Band to plot.
    ax : `matplotlib.axes.Axes`, optional
        Axis to draw on; a new figure is made if omitted.
    title : `str`, optional
        Axis title.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        The axis drawn on.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    band_curve = curve[curve["band"] == band]
    axis = band_curve["axis"].iloc[0] if len(band_curve) else "snr"
    for (epoch, method), group in band_curve.groupby(["epoch", "method"]):
        group = group.sort_values("x")
        ax.plot(group["x"], group["completeness"],
                linestyle=_METHOD_STYLE.get(method, "-"), marker=_METHOD_MARKER.get(method, "o"),
                color=_EPOCH_COLOR.get(epoch, "#555555"), markersize=4, linewidth=2,
                label=f"{method} {epoch}")
    if axis == "mag":
        ax.set_xlabel("truth AB magnitude")
    else:
        ax.set_xlabel("truth peak S/N")
        ax.set_xscale("log")
    ax.set_ylabel("completeness")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize="small", ncol=2)
    ax.set_title(title or f"{band} band")
    return ax


def match_distance_histogram(truth, edges, normalize=True):
    """Match-distance distribution per method.

    Parameters
    ----------
    truth : `pandas.DataFrame`
        The truth-side table from `evaluate`; only detected peaks contribute.
    edges : `numpy.ndarray`
        Distance bin edges, in pixels.
    normalize : `bool`, optional
        Return each method's fraction of matches per bin (so methods with
        different detection counts are comparable) rather than raw counts.

    Returns
    -------
    hist : `pandas.DataFrame`
        Columns ``method``, ``distance`` (bin centre) and ``value`` (fraction
        if ``normalize``, else count).
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    detected = truth[truth["detected"]]
    for method, group in detected.groupby("method"):
        counts, _ = np.histogram(group["match_dist"].to_numpy(float), bins=edges)
        values = counts / counts.sum() if normalize and counts.sum() else counts.astype(float)
        for centre, value in zip(centres, values):
            rows.append(dict(method=method, distance=centre, value=float(value)))
    return pd.DataFrame(rows)


def split_examples(peaks, max_extent=40.0, min_peaks=4, limit=12):
    """Rank clique-split peak groups for a worked image inspection.

    A detection peak with ``structure_peaks > 1`` was split off a larger
    connected structure; this groups such peaks by ``structure_id`` and keeps
    the compact, many-peak ones, which are the clearest cases to look at to see
    whether a group is tracking real blended structure or over-splitting it.

    Parameters
    ----------
    peaks : `pandas.DataFrame`
        A new-detection peak table with ``x``, ``y``, ``structure_id`` and
        ``structure_peaks`` (e.g. ``NewDetectionReader.peaks`` output for one
        patch).
    max_extent : `float`, optional
        Keep only structures whose peaks span at most this many pixels in x and
        y, so a stamp of that size shows the whole group.
    min_peaks : `int`, optional
        Keep only structures with at least this many peaks.
    limit : `int`, optional
        Return at most this many groups, most peaks first.

    Returns
    -------
    examples : `pandas.DataFrame`
        One row per structure: ``structure_id``, ``n_peaks``, ``x``, ``y`` (the
        group centre) and ``extent`` (the larger of the x, y spans).
    """
    split = peaks[peaks["structure_peaks"] > 1]
    rows = []
    for structure_id, group in split.groupby("structure_id"):
        ex = max(group["x"].max() - group["x"].min(), group["y"].max() - group["y"].min())
        if len(group) >= min_peaks and ex <= max_extent:
            rows.append(dict(structure_id=structure_id, n_peaks=len(group),
                             x=group["x"].mean(), y=group["y"].mean(), extent=ex))
    out = pd.DataFrame(rows)
    return out.sort_values("n_peaks", ascending=False).head(limit).reset_index(drop=True)
