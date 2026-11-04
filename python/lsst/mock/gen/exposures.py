"""Read a coadd written by `~mock_lsst_image_generation.image_pipeline`
into afw exposures and astropy tables.
"""

__all__ = ["MockCoadd", "read_mock_coadd", "make_exposure", "make_exposures", "read_truth_tables"]

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
from lsst.meas.algorithms import KernelPsf


@dataclass
class MockCoadd:
    """The pixel products and settings of one saved mock coadd.

    The pixel arrays are memory maps of the ``.npy`` files, with shape
    ``(band, y, x)``.
    """

    name: str
    """Name of the catalogue the coadd was rendered from."""
    key: str
    """Coadd key in the manifest, e.g. ``"1y_moffat"``."""
    directory: Path
    """Directory holding the coadd's files."""
    bands: list[str]
    """Bands in the order of the first array axis."""
    origin: tuple[int, int]
    """Frame pixel ``(x, y)`` of array element ``[:, 0, 0]``."""
    pixel_scale: float
    """Pixel scale in arcsec."""
    signal: np.ndarray
    """Image in nJy per pixel."""
    variance: np.ndarray
    """Per-pixel variance in nJy^2."""
    mask: np.ndarray
    """Mask bits, named by ``mask_planes``."""
    psf: np.ndarray
    """PSF image of each band, centred on the middle pixel."""
    mask_planes: dict[str, int]
    """Bit value of each named mask plane in ``mask``."""
    settings: dict
    """The coadd's manifest entry: depth, visits and PSF per band."""

    @property
    def bbox(self) -> geom.Box2I:
        """Pixel bounding box in frame coordinates (`lsst.geom.Box2I`)."""
        _, height, width = self.signal.shape
        return geom.Box2I(geom.Point2I(*self.origin), geom.Extent2I(width, height))


def read_mock_coadd(image_dir: str | Path, coadd: str | None = None) -> MockCoadd:
    """Read one saved coadd of a rendered mock catalogue.

    Parameters
    ----------
    image_dir : `str` or `pathlib.Path`
        Directory written for one catalogue, ``<out_dir>/<name>``, holding
        ``base_meta.json`` and ``coadd_manifest.json``.
    coadd : `str`, optional
        Key of the coadd to read. May be omitted when exactly one coadd in
        the manifest was saved.

    Returns
    -------
    coadd : `MockCoadd`
        The coadd's pixels and settings.

    Raises
    ------
    ValueError
        Raised if ``coadd`` is not a saved coadd, or is omitted and the
        directory does not hold exactly one.
    """
    directory = Path(image_dir)
    meta = json.loads((directory / "base_meta.json").read_text())
    combos = json.loads((directory / "coadd_manifest.json").read_text())["combos"]
    name = meta["split"]
    saved = [key for key in combos if (directory / f"{name}_{key}_signal.npy").exists()]
    if coadd is None:
        if len(saved) != 1:
            raise ValueError(f"{directory} has saved coadds {saved}; choose one")
        coadd = saved[0]
    elif coadd not in saved:
        raise ValueError(f"coadd {coadd!r} is not saved in {directory}; saved coadds are {saved}")

    def load(product):
        return np.load(directory / f"{name}_{coadd}_{product}.npy", mmap_mode="r")

    return MockCoadd(
        name=name,
        key=coadd,
        directory=directory,
        bands=list(meta["bands"]),
        origin=(int(meta["origin"][0]), int(meta["origin"][1])),
        pixel_scale=float(meta["ps"]),
        signal=load("signal"),
        variance=load("variance"),
        mask=load("mask"),
        psf=load("psf"),
        mask_planes=dict(meta["mask_planes"]),
        settings=combos[coadd],
    )


def make_exposure(coadd: MockCoadd, band: str, wcs: afwGeom.SkyWcs | None = None) -> afwImage.ExposureF:
    """Make an exposure from one band of a mock coadd.

    Parameters
    ----------
    coadd : `MockCoadd`
        Coadd to convert.
    band : `str`
        Band to convert.
    wcs : `lsst.afw.geom.SkyWcs`, optional
        WCS to attach, normally that of the tract the coadd is assigned to.

    Returns
    -------
    exposure : `lsst.afw.image.ExposureF`
        Exposure in nJy with the coadd's variance, mask and PSF. Its origin
        is the frame pixel of the first array element, so the catalogue's
        ``x_pix`` and ``y_pix`` are its pixel coordinates.

    Raises
    ------
    ValueError
        Raised if the PSF image has an even size, so has no centre pixel.
    """
    index = coadd.bands.index(band)
    psf = np.asarray(coadd.psf[index], dtype=np.float64)
    if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
        raise ValueError(f"the PSF image must have odd dimensions, got {psf.shape}")

    # A mask copies the plane registry when it is made, so the planes are
    # registered first.
    for plane in coadd.mask_planes:
        afwImage.Mask.addMaskPlane(plane)
    maskedImage = afwImage.MaskedImageF(coadd.bbox)
    maskedImage.image.array[:] = coadd.signal[index]
    maskedImage.variance.array[:] = coadd.variance[index]
    for plane, bit in coadd.mask_planes.items():
        maskedImage.mask.array[(coadd.mask[index] & bit) != 0] |= maskedImage.mask.getPlaneBitMask(plane)

    exposure = afwImage.ExposureF(maskedImage)
    exposure.setPsf(KernelPsf(afwMath.FixedKernel(afwImage.ImageD(psf))))
    # The pixels are already in nJy.
    exposure.setPhotoCalib(afwImage.PhotoCalib(1.0))
    exposure.setFilter(afwImage.FilterLabel(band=band))
    if wcs is not None:
        exposure.setWcs(wcs)

    settings = coadd.settings
    metadata = exposure.getMetadata()
    metadata["MOCK_CATALOGUE"] = coadd.name
    metadata["MOCK_COADD"] = coadd.key
    metadata["MOCK_PSF_MODEL"] = settings.get("psf_model", "gaussian")
    metadata["MOCK_N_VISIT"] = int(settings["n_visit"][band])
    metadata["MOCK_FWHM"] = float(settings["fwhm_arcsec"][band])
    if settings.get("moffat_beta"):
        metadata["MOCK_MOFFAT_BETA"] = float(settings["moffat_beta"][band])
    return exposure


def make_exposures(coadd: MockCoadd, wcs: afwGeom.SkyWcs | None = None) -> dict[str, afwImage.ExposureF]:
    """Make an exposure from every band of a mock coadd.

    Parameters
    ----------
    coadd : `MockCoadd`
        Coadd to convert.
    wcs : `lsst.afw.geom.SkyWcs`, optional
        WCS to attach to every exposure.

    Returns
    -------
    exposures : `dict` [`str`, `lsst.afw.image.ExposureF`]
        Exposure of each band, in the coadd's band order.
    """
    return {band: make_exposure(coadd, band, wcs) for band in coadd.bands}


def to_astropy(table: pd.DataFrame) -> Table:
    """Convert a truth table to astropy, giving text columns a string dtype.

    Text columns read from CSV hold `float` NaN where a row has no value,
    which parquet cannot store in a string column.
    """
    table = table.copy()
    for column in table.select_dtypes(include="object").columns:
        table[column] = table[column].fillna("").astype(str)
    return Table.from_pandas(table)


def read_truth_tables(coadd: MockCoadd) -> dict[str, Table]:
    """Read the truth tables written for a coadd by ``make_truth_peaks.py``.

    Parameters
    ----------
    coadd : `MockCoadd`
        Coadd whose tables to read.

    Returns
    -------
    tables : `dict` [`str`, `astropy.table.Table`]
        ``"truth"`` (one row per component) and ``"peaks"`` (one row per
        noise-free local maximum), for those that exist on disk.
    """
    tables = {}
    for kind in ("truth", "peaks"):
        path = coadd.directory / f"{coadd.name}_{coadd.key}_{kind}.csv"
        if path.exists():
            tables[kind] = to_astropy(pd.read_csv(path, low_memory=False))
    return tables
