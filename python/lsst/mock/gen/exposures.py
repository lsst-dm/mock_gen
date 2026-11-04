"""Turn a rendered mock coadd into an afw exposure, and a truth DataFrame into an astropy table."""

__all__ = ["make_exposure", "to_astropy"]

import numpy as np
import pandas as pd
from astropy.table import Table

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.meas.algorithms import KernelPsf


def make_exposure(signal, variance, mask, psf, band, mask_planes, bbox,
                  wcs=None, metadata=None):
    """Make an exposure from one band's rendered mock coadd arrays.

    Parameters
    ----------
    signal : `numpy.ndarray`
        Image in nJy per pixel, shape ``(y, x)``.
    variance : `numpy.ndarray`
        Per-pixel variance in nJy^2, same shape as ``signal``.
    mask : `numpy.ndarray`
        Mask bits named by ``mask_planes``, same shape as ``signal``.
    psf : `numpy.ndarray`
        PSF image centred on the middle pixel; must have odd dimensions.
    band : `str`
        Band label for the exposure's filter.
    mask_planes : `dict` [`str`, `int`]
        Bit value of each named mask plane in ``mask``.
    bbox : `lsst.geom.Box2I`
        Pixel bounding box to give the exposure.
    wcs : `lsst.afw.geom.SkyWcs`, optional
        WCS to attach.
    metadata : `dict`, optional
        Header keyword/value pairs to add to the exposure metadata.

    Returns
    -------
    exposure : `lsst.afw.image.ExposureF`
        Exposure in nJy with the given variance, mask and PSF, a unit
        `~lsst.afw.image.PhotoCalib` (the pixels are already in nJy) and the
        band's filter label.

    Raises
    ------
    ValueError
        Raised if the PSF image has an even dimension, so has no centre pixel.
    """
    psf = np.asarray(psf, dtype=np.float64)
    if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
        raise ValueError(f"the PSF image must have odd dimensions, got {psf.shape}")

    # A mask copies the plane registry when it is made, so the planes are registered first.
    for plane in mask_planes:
        afwImage.Mask.addMaskPlane(plane)
    maskedImage = afwImage.MaskedImageF(bbox)
    maskedImage.image.array[:] = signal
    maskedImage.variance.array[:] = variance
    for plane, bit in mask_planes.items():
        maskedImage.mask.array[(mask & bit) != 0] |= maskedImage.mask.getPlaneBitMask(plane)

    exposure = afwImage.ExposureF(maskedImage)
    exposure.setPsf(KernelPsf(afwMath.FixedKernel(afwImage.ImageD(psf))))
    exposure.setPhotoCalib(afwImage.PhotoCalib(1.0))
    exposure.setFilter(afwImage.FilterLabel(band=band))
    if wcs is not None:
        exposure.setWcs(wcs)
    if metadata:
        header = exposure.getMetadata()
        for key, value in metadata.items():
            header[key] = value
    return exposure


def to_astropy(table: pd.DataFrame) -> Table:
    """Convert a truth or catalogue table to astropy, giving text columns a string dtype.

    Text columns hold `float` NaN where a row has no value, which parquet cannot
    store in a string column.
    """
    table = table.copy()
    for column in table.select_dtypes(include="object").columns:
        table[column] = table[column].fillna("").astype(str)
    return Table.from_pandas(table)
