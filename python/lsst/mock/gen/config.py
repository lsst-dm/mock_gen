"""Config for the mock generator tasks: the knobs worth varying between runs.

The mock model in ``mock_lsst_image_generation`` has a few hundred settings
(``CONFIG``, ``IMAGE_CONFIG``, ``STAR_CONFIG``, ``PHYS``). Most are calibrated
model internals that should keep their defaults. The fields here are the subset
worth setting per run: the scene content (`CatalogueConfig`) and the depth, PSF
and noise (`ImageConfig`). ``build_cfg`` turns each into the partial dict that
``build_mock_catalogue`` / ``render_single_coadd`` merge over the model
defaults, so every unexposed setting keeps its upstream value. To expose another
knob, add a field and map it in ``build_cfg``.
"""

__all__ = ["CatalogueConfig", "ImageConfig"]

import lsst.pex.config as pexConfig

from mock_lsst_image_generation.config import CONFIG, IMAGE_CONFIG, STAR_CONFIG


class CatalogueConfig(pexConfig.Config):
    """Scene-content settings for `~lsst.mock.gen.generate.MockCatalogGenTask`.

    The seed is the tract id, not a field. ``build_cfg`` returns a partial
    ``CONFIG`` override; unset model settings keep their defaults.
    """

    npix = pexConfig.Field(dtype=int, default=CONFIG["npix"], doc="Side of the square frame in pixels.")
    pixscale = pexConfig.Field(dtype=float, default=CONFIG["pixscale"], doc="Pixel scale in arcsec.")

    stars = pexConfig.Field(dtype=bool, default=CONFIG["stars"], doc="Add bright stars.")
    bcgs = pexConfig.Field(dtype=bool, default=CONFIG["bcgs"], doc="Add BCG-like core-Sersic sources.")
    lsb_galaxies = pexConfig.Field(dtype=bool, default=CONFIG["lsb_galaxies"],
                                   doc="Add extended dwarfs, UDGs and almost-dark galaxies.")
    sfregions = pexConfig.Field(dtype=bool, default=CONFIG["sfregions"], doc="Add star-forming regions.")
    tidal_streams = pexConfig.Field(dtype=bool, default=CONFIG["tidal_streams"],
                                    doc="Add tidal bridges between interacting pairs.")
    hubble_types = pexConfig.Field(dtype=bool, default=CONFIG["hubble_types"],
                                   doc="Draw Hubble types with their arm / bar / clump structure.")

    star_density_per_deg2 = pexConfig.Field(dtype=float, default=STAR_CONFIG["density_per_deg2"],
                                            doc="Field star density per square degree.")
    star_bright_per_deg2 = pexConfig.Field(dtype=float, default=STAR_CONFIG["bright_per_deg2"],
                                           doc="Extra bright stars per square degree (spikes and saturation).")
    bcg_per_deg2 = pexConfig.Field(dtype=float, default=CONFIG["bcg_per_deg2"],
                                   doc="BCG-like sources per square degree.")
    lsb_extended_dirr_per_deg2 = pexConfig.Field(
        dtype=float, default=CONFIG["lsb_populations"]["extended_dirr"]["per_deg2"],
        doc="Extended dwarf irregulars per square degree.")
    lsb_udg_per_deg2 = pexConfig.Field(dtype=float, default=CONFIG["lsb_populations"]["udg"]["per_deg2"],
                                       doc="Ultra-diffuse galaxies per square degree.")
    lsb_almost_dark_per_deg2 = pexConfig.Field(
        dtype=float, default=CONFIG["lsb_populations"]["almost_dark"]["per_deg2"],
        doc="Almost-dark galaxies per square degree.")

    tidal_target_field = pexConfig.Field(dtype=float, default=CONFIG["tidal_target_field"],
                                         doc="Fraction of field galaxies given a tidal bridge.")
    tidal_target_filament = pexConfig.Field(dtype=float, default=CONFIG["tidal_target_filament"],
                                            doc="Fraction of filament galaxies given a tidal bridge.")
    tidal_target_cluster = pexConfig.Field(dtype=float, default=CONFIG["tidal_target_cluster"],
                                           doc="Fraction of cluster galaxies given a tidal bridge.")

    web_f_cluster = pexConfig.Field(dtype=float, default=CONFIG["web_f_cluster"],
                                    doc="Fraction of galaxies on cluster sites of the cosmic web.")
    web_f_filament = pexConfig.Field(dtype=float, default=CONFIG["web_f_filament"],
                                     doc="Fraction of galaxies on filament sites of the cosmic web.")

    z_min = pexConfig.Field(dtype=float, default=CONFIG["z_min"], doc="Minimum galaxy redshift.")
    z_max = pexConfig.Field(dtype=float, default=CONFIG["z_max"], doc="Maximum galaxy redshift.")
    logm_min = pexConfig.Field(dtype=float, default=CONFIG["logm_min"], doc="Minimum log10 stellar mass.")
    logm_max = pexConfig.Field(dtype=float, default=CONFIG["logm_max"], doc="Maximum log10 stellar mass.")

    render_mu_r_max = pexConfig.Field(dtype=float, default=CONFIG["render_mu_r_max"],
                                      doc="Faintest mean r-band surface brightness worth rendering (mag/arcsec^2).")
    render_mag_r_max = pexConfig.Field(dtype=float, default=CONFIG["render_mag_r_max"], optional=True,
                                       doc="Faintest r-band magnitude worth rendering (None keeps every source).")

    def build_cfg(self):
        """Partial ``CONFIG`` override for ``build_mock_catalogue``'s ``cfg``.

        Returns
        -------
        cfg : `dict`
            The exposed settings in the model's own ``CONFIG`` shape. The LSB
            rates fold into an ``lsb_populations`` copy; star settings are
            separate (`build_star_cfg`); every unset model key is left out so it
            keeps its default.
        """
        lsb = CONFIG["lsb_populations"]
        lsb_populations = {
            "extended_dirr": dict(lsb["extended_dirr"], per_deg2=self.lsb_extended_dirr_per_deg2),
            "udg": dict(lsb["udg"], per_deg2=self.lsb_udg_per_deg2),
            "almost_dark": dict(lsb["almost_dark"], per_deg2=self.lsb_almost_dark_per_deg2),
        }
        return dict(
            npix=self.npix, pixscale=self.pixscale,
            stars=self.stars, bcgs=self.bcgs, lsb_galaxies=self.lsb_galaxies, sfregions=self.sfregions,
            tidal_streams=self.tidal_streams, hubble_types=self.hubble_types,
            bcg_per_deg2=self.bcg_per_deg2, lsb_populations=lsb_populations,
            tidal_target_field=self.tidal_target_field, tidal_target_filament=self.tidal_target_filament,
            tidal_target_cluster=self.tidal_target_cluster,
            web_f_cluster=self.web_f_cluster, web_f_filament=self.web_f_filament,
            z_min=self.z_min, z_max=self.z_max, logm_min=self.logm_min, logm_max=self.logm_max,
            render_mu_r_max=self.render_mu_r_max, render_mag_r_max=self.render_mag_r_max,
        )

    def build_star_cfg(self):
        """Partial ``STAR_CONFIG`` override for ``build_mock_catalogue``'s ``star_cfg``.

        Returns
        -------
        star_cfg : `dict`
            The exposed star rates in the model's ``STAR_CONFIG`` shape; unset
            star settings keep their defaults.
        """
        return dict(density_per_deg2=self.star_density_per_deg2, bright_per_deg2=self.star_bright_per_deg2)


class ImageConfig(pexConfig.Config):
    """Depth, PSF and noise settings for `~lsst.mock.gen.generate.MockCoaddGenTask`.

    The epoch and noise realisation come from the patch id, not fields.
    ``build_cfg`` returns a partial ``IMAGE_CONFIG`` override.
    """

    psf_model = pexConfig.ChoiceField(
        dtype=str, default="moffat",
        allowed={"gaussian": "Gaussian core, halo and spikes on stars only",
                 "moffat": "Moffat core with halo and spikes shared by every source"},
        doc="PSF model shared by the coadd's bands.")
    canvas_npix = pexConfig.Field(
        dtype=int, default=3000,
        doc="Side of each coadd in pixels; normally the catalogue's npix.")

    depth_10yr = pexConfig.DictField(keytype=str, itemtype=float, default=dict(IMAGE_CONFIG["depth_10yr"]),
                                     doc="10-year 5-sigma point-source depth per band (AB).")
    visits_10yr = pexConfig.DictField(keytype=str, itemtype=int, default=dict(IMAGE_CONFIG["visits_10yr"]),
                                      doc="10-year visit count per band.")
    nominal_fwhm = pexConfig.DictField(keytype=str, itemtype=float, default=dict(IMAGE_CONFIG["nominal_fwhm"]),
                                       doc="Nominal 10-year median-seeing PSF FWHM per band (arcsec).")
    moffat_fwhm = pexConfig.DictField(keytype=str, itemtype=float, default=dict(IMAGE_CONFIG["moffat_fwhm"]),
                                      doc="Moffat-model FWHM per band (arcsec).")
    moffat_beta = pexConfig.DictField(keytype=str, itemtype=float, default=dict(IMAGE_CONFIG["moffat_beta"]),
                                      doc="Moffat-model index per band.")

    base_fwhm = pexConfig.Field(dtype=float, default=IMAGE_CONFIG["base_fwhm"],
                                doc="Narrow base-render PSF FWHM (arcsec); must be narrower than every target.")
    fwhm_ratio_jitter = pexConfig.Field(dtype=float, default=IMAGE_CONFIG["fwhm_ratio_jitter"],
                                        doc="Fractional per-band random jitter on the coadd FWHM.")
    moffat_beta_jitter = pexConfig.Field(dtype=float, default=IMAGE_CONFIG["moffat_beta_jitter"],
                                         doc="Fractional per-band random jitter on the Moffat index.")

    def build_cfg(self, bands):
        """Partial ``IMAGE_CONFIG`` override for ``render_single_coadd``.

        Parameters
        ----------
        bands : `list` [`str`]
            Bands to render, in output order.

        Returns
        -------
        cfg : `dict`
            The exposed settings in the model's ``IMAGE_CONFIG`` shape, plus the
            band list and canvas size; unset keys keep their defaults.
        """
        return dict(
            bands=list(bands), psf_model=self.psf_model, canvas_npix=self.canvas_npix,
            depth_10yr=dict(self.depth_10yr), visits_10yr=dict(self.visits_10yr),
            nominal_fwhm=dict(self.nominal_fwhm), moffat_fwhm=dict(self.moffat_fwhm),
            moffat_beta=dict(self.moffat_beta), base_fwhm=self.base_fwhm,
            fwhm_ratio_jitter=self.fwhm_ratio_jitter, moffat_beta_jitter=self.moffat_beta_jitter,
        )
