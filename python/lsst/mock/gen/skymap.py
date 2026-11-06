"""A minimal skymap for the mock repository.

The mock frames carry no real sky geometry: a tract id is a catalogue seed and a
patch id encodes a noise realisation and epoch (see `~lsst.mock.gen.generate`).
Most tasks do not need a skymap, but some coadd tasks (e.g.
`~lsst.pipe.tasks.mergeDetections.MergeDetectionsTask`) take a ``SkyMap`` dataset
as a required input and look up a tract by its id. `MockSkyMap` satisfies that:
it exposes an explicit list of tract ids, all sharing one COSMOS-centred TAN
tract whose patch 0 is frame pixels ``[0, npix)``, matching the coadds. A
`~lsst.skymap.DiscreteSkyMap` cannot be used because its tract id is the
sequential index, so it cannot hold ids like 9811.
"""

__all__ = ["MockSkyMap", "MockSkyMapConfig"]

import struct

import lsst.geom as geom
import lsst.pex.config as pexConfig
from lsst.skymap import BaseSkyMap
from lsst.skymap.cachingSkyMap import CachingSkyMap
from lsst.skymap.patchInfo import PatchInfo
from lsst.skymap.tractInfo import ExplicitTractInfo

# The mock frames are built from COSMOS2025, so the tract sits on the COSMOS field.
_COSMOS_RA = 150.1191
_COSMOS_DEC = 2.2058


class _MockTractInfo(ExplicitTractInfo):
    """Tract whose every patch is the one frame at the image origin.

    A patch id selects a noise realisation and epoch, not a position on the sky,
    and the coadd for every patch is written at pixel origin ``(0, 0)``. The base
    `~lsst.skymap.TractInfo` instead lays patches out on a grid, so patch id 1
    gets a bbox at ``(npix, 0)`` and so on. A task that reads the patch bbox and
    the patch image together (e.g. `MergeDetectionsTask`, which allocates a mask
    over ``getOuterBBox()`` and inserts the detection footprints) then sees the
    footprints, at the origin, fall outside the bbox for any patch but 0. Return
    the frame bbox for every valid patch id so the two always agree.
    """

    def getPatchInfo(self, index):
        # Validate the id and reuse the sequential index and WCS the base
        # computes; only the bbox is moved to the frame origin. patchBorder is 0,
        # so the base inner bbox already has the frame dimensions.
        info = super().getPatchInfo(index)
        bbox = geom.Box2I(geom.Point2I(0, 0), info.getInnerBBox().getDimensions())
        return PatchInfo(index=info.getIndex(), innerBBox=bbox, outerBBox=bbox,
                         sequentialIndex=info.getSequentialIndex(), tractWcs=info.getWcs())


class MockSkyMapConfig(BaseSkyMap.ConfigClass):
    """Config for `MockSkyMap`: an explicit list of tract ids over one frame."""

    tractIds = pexConfig.ListField(
        dtype=int, default=[0], doc="Tract ids to expose; each shares the same frame geometry.")
    npix = pexConfig.Field(dtype=int, default=3000, doc="Side of the frame (one patch) in pixels.")
    nPatches = pexConfig.Field(dtype=int, default=11, doc="Patches per axis; must exceed the largest patch id.")
    ra = pexConfig.Field(dtype=float, default=_COSMOS_RA, doc="Tract centre RA in degrees.")
    dec = pexConfig.Field(dtype=float, default=_COSMOS_DEC, doc="Tract centre Dec in degrees.")

    def setDefaults(self):
        super().setDefaults()
        self.pixelScale = 0.2  # arcsec, matching the mock coadds (IMAGE_CONFIG pixscale)
        self.projection = "TAN"
        self.tractBuilder["legacy"].patchInnerDimensions = [self.npix, self.npix]
        self.tractBuilder["legacy"].patchBorder = 0
        self.tractOverlap = 0.0


class MockSkyMap(CachingSkyMap):
    """A skymap of tracts with explicit ids sharing one COSMOS-centred frame.

    Parameters
    ----------
    config : `MockSkyMapConfig`
        Configuration, including the tract ids and frame size.
    version : `int`, optional
        Software version, for the hash.
    """

    ConfigClass = MockSkyMapConfig

    def __init__(self, config, version=0):
        # patchInnerDimensions is set from npix in the config, but a config built field by field (not via its
        # defaults) may not have followed npix; pin it here so the patch grid always matches npix.
        config.tractBuilder["legacy"].patchInnerDimensions = [config.npix, config.npix]
        config.tractBuilder["legacy"].patchBorder = 0
        super().__init__(len(config.tractIds), config, version)
        self._id_to_index = {tract_id: index for index, tract_id in enumerate(config.tractIds)}

    def generateTract(self, index):
        """Generate the `~lsst.skymap.ExplicitTractInfo` for the tract at a sequential ``index``."""
        center = geom.SpherePoint(self.config.ra, self.config.dec, geom.degrees)
        # A radius covering the whole patch grid, so patches 0 .. nPatches-1 exist.
        radius = (self.config.nPatches * self.config.npix) * self.config.pixelScale / 3600 / 2 * geom.degrees
        wcs = self._wcsFactory.makeWcs(crPixPos=geom.Point2D(0, 0), crValCoord=center)
        return _MockTractInfo(self.config.tractIds[index], self._tractBuilder, center,
                              radius, self.config.tractOverlap * geom.degrees, wcs)

    def __iter__(self):
        """Iterate tracts by sequential position (used by `register`)."""
        for index in range(self._numTracts):
            yield super().__getitem__(index)

    def __getitem__(self, tract_id):
        """Get a tract by its id (not its sequential position)."""
        return super().__getitem__(self._id_to_index[tract_id])

    def updateSha1(self, sha1):
        """Fold the tract ids and frame size into the uniqueness hash."""
        sha1.update(struct.pack("<qq", self.config.npix, self.config.nPatches))
        sha1.update(struct.pack(f"<{len(self.config.tractIds)}q", *self.config.tractIds))
