
# This file can calculate the whole spectrum for the real observation. But for simulation, it will calculate only up to four wavelength slices to save time.
#
import math
import numpy as np
import scipy.interpolate as interpolate
import marvin.tools.rss as rss
from marvin import config
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from astropy.io import fits
from pydl.pydlutils.sdss import sdss_flagval
from scipy import sparse
import time
import os

# Set Marvin configuration so it gets everything local
config.setRelease('MPL-6')
config.mode = 'local'
config.download = True

# Rough grid length in arcsec to use for each IFU size
gridsize = {7: 12, 19: 17, 37: 22, 61: 27, 91: 32, 127: 36}


class Reconstruct(object):
    """Base class for reconstruction of cubes from RSS files
       Should not be used directly.

    Attributes:
    ----------

    plate : int, np.int32
        plate number

    ifu : int, np.int32
        IFU number

    nfiber : int
        number of fibers

    release : str
        data release (default 'MPL-5')

    rss : RSS object
        Marvin RSS output

    waveindex : int, np.int32
        indices of wavelengths to reconstruct (default None)

    Methods:
    -------

    set_rss() : Acquire the RSS data

    set_image_grid(dimage) : Set up the spatial grid for the cube

    set_kernel(fwhm) : Create the kernel estimation from accessing to data base

    set_flux_rss() : Set the flux used for reconstruction from RSS

    set_flux_psf(xcen=0., ycen=0.,alpha=1,noise=None) : Set the flux used for reconstruction to a PSF

    set_weights() : Set the weights for mapping spectra to cube

    create_weights() : Calculate the weights for mapping spectra to cube

    normalize_weights(w) : Normalize the weights

    set_cube() : Calculates and sets cube for both RSS and simulation

    calculate_cube(flux,flux_ivar,flux_mask): Calculate the result for given flux, flux_ivar and flux_mask

    covar() : Calculate covariance matrix for a slice of the cube

    mask(): Calculate mask matrix for a slice of the cube

    plot_slice() : Plots a slice of the cube

    FWHM(xi=None, yi=None, PSF=None, xcen=0, ycen=0): calculate FWHM for given reconstructed image

    Notes:
    ------

    Additional attributes are set by the methods, as documented.

    Unless waveindex is set, uses all wavelengths.

    Typical usage would be (for a limited number of wavelengths):

     import reconstruction
     r = reconstruction.Reconstruct(plate=8485, ifu=3701, waveindex=[1000, 2000])
     r.set_rss() # Gets and reads in the RSS file (sets .rss attribute)
     r.set_image_grid() # Creates the output image spatial grid
     r.set_kernel() # Sets the kernel for every wavelength and exposure
     r.set_flux_rss() # Sets up to use the RSS fluxes (sets .flux, .flux_ivar)
     r.set_flux_psf(xcen=0.5, ycen=1.5, alpha=1,noise=5) # Puts fake data into PSF fluxes (sets .flux_psf, .flux_psf_ivar)
     r.set_weights() # Sets the weights (sets .weights)
     r.set_cube() # Sets the weights (sets .cube, .cube_ivar .cube_mask)
     r.plot_slice(0)

"""

    def __init__(self, plate=None, ifu=None, release='MPL-5', waveindex=None):
        self.plate = plate
        self.ifu = ifu
        self.plateifu = "{plate}-{ifu}".format(plate=self.plate, ifu=self.ifu)
        self.release = release
        self.nfiber = int(self.ifu / 100)
        self.rss = None
        self.waveindex = waveindex
        if (self.waveindex is not None):
            self.waveindex = self._arrayify_int32(self.waveindex)
        return

    def _arrayify_int32(self, quantity):
        """Cast quantity as ndarray of numpy.int32"""
        try:
            length = len(quantity)
        except TypeError:
            length = 1
        return np.zeros(length, dtype=np.int32) + quantity

    def set_rss(self):
        """Acquire the RSS data and set related attributes

        Parameters:
        ----------

        Notes:
        -----

        Sets attributes:

         .rss - Marvin RSS object
         .nExp - Number of exposures
         .xpos - X positions of each fiber [nExp * nfiber, nWave]
         .ypos - Y positions of each fiber [nExp * nfiber, nWave]
         .wave - Selected wavelength grid [nWave]
         .nWave - Size of wavelength grid
         .obsinfo - Observing information object
         .fwhm0 - FWHM of seeing at guider wavelength (5400 Angstroms) [nExp]
         .fwhm - FWHM at each wavelength [nWave, nExp]
"""
        self.rss = rss.RSS(plateifu=self.plateifu, release=self.release)
        self.nExp = self.rss.data[0].header['NEXP']
        self.xpos = self.rss.data['XPOS'].data
        self.ypos = self.rss.data['YPOS'].data

        self.wave = self.rss.data['WAVE'].data
        self.nWave = self.rss.data['FLUX'].shape[1]
        gpivot = 4702.50
        rpivot = 6176.58
        ipivot = 7496.12
        zpivot = 8946.71
        self.bbwave = [gpivot, rpivot, ipivot, zpivot]
        self.bbindex = []
        for wave in self.bbwave:
            self.bbindex.append(min(range(len(self.wave)), key=lambda i: abs(self.wave[i] - wave)))
        # Use waveindex if that was set
        if (self.waveindex is not None):
            self.nWave = len(self.waveindex)
            self.wave = self.wave[self.waveindex]
            self.xpos = self.xpos[:, self.waveindex]
            self.ypos = self.ypos[:, self.waveindex]

        # Set FWHM values as a function of wavelength
        self.obsinfo = self.rss.data['OBSINFO'].data
        self.fwhm0 = (self.obsinfo.field('SEEING') *
                      self.obsinfo.field('PSFFAC'))

        lambda0 = 5400.
        self.fwhm = [[self.fwhm0[i] * math.pow(lambda0 / self.wave[j], 1. / 5.)
                      for i in range(self.fwhm0.shape[0])]
                     for j in range(self.nWave)]
        self.fwhm = np.array(self.fwhm)
        #         self.fwhm = np.ones_like(self.fwhm)*self.fwhm.mean()

        return

    def _create_grid(self, d=None):
        """Create a grid (used for image and kernel)"""
        rough_length = gridsize[self.nfiber]
        nside = ((np.int32(np.ceil(rough_length / d)) // 2) * 2 + 1)
        length = nside * d
        xmin = -0.5 * length
        xmax = 0.5 * length
        ymin = -0.5 * length
        ymax = 0.5 * length
        xi = np.linspace(xmin + 0.5 * d, xmax - 0.5 * d, nside)
        yi = np.linspace(ymin + 0.5 * d, ymax - 0.5 * d, nside)
        x2i, y2i = np.meshgrid(xi, yi)
        return (nside, length, x2i, y2i, xi, yi)

    def set_image_grid(self, dimage=0.75):
        """Create the image output grid

        Parameters:
        ----------

        dimage : float, np.float32
            pixel scale for output grid in arcsec (default: 0.75)

        Notes:
        -----

        Sets attributes:

         .dimage - pixel scale for output grid in arcsec
         .conversion - factor to multiply per fiber units by to get
                       per pixel units
         .nside - number of pixels on a side for the image grid
         .nimage - number of pixels total in image
         .length - length of image grid in arcsec (based on outer pixel edges)
         .x2i - 2-D array of X positions
         .y2i - 2-D array of Y positions
         .xi - 1-D array of X positions
         .yi - 1-D array of Y positions
         .xmin - outer edge of lowest X pixel
         .ymin - outer edge of lowest Y pixel
         .xmax - outer edge of highest X pixel
         .ymax - outer edge of highest Y pixel
"""
        self.dimage = dimage
        self.conversion = self.dimage ** 2 / np.pi
        (self.nside, self.length, self.x2i, self.y2i, self.xi, self.yi) = self._create_grid(self.dimage)
        self.nimage = self.nside ** 2
        self.xmin = -0.5 * self.length
        self.xmax = 0.5 * self.length
        self.ymin = -0.5 * self.length
        self.ymax = 0.5 * self.length
        return

    def set_kernel(self, dkernel=0.05):
        """Set the kernel for each wavelength and exposure

        Parameters:
        ----------

        dkernel : float, np.float32
            pixel scale for kernel grid in arcsec (default: 0.05)

        Notes:
        -----

        Sets attributes:

         .x2k - 2-D array of X positions of kernel grid
         .y2k - 2-D array of Y positions of kernel grid
         .xkernel - 1-D array of X positions of kernel grid
         .ykernel - 1-D array of Y positions of kernel grid
         .nkernel - number of pixels on a side of kernel image
         .kernelbase - 3D array of kernel data used to access kernel value [250,nkernelbase,nkernelbase]
         .nkernelbase - number of pixels on a side of base kernel slice
         .kernel_radial - the value of kernel at given radius
"""
        self.dkernel = dkernel
        (self.nkernel, length, self.x2k, self.y2k, self.xkernel, self.ykernel) = self._create_grid(self.dkernel)

        filename = os.path.join(os.getenv('RECONSTRUCTION_DIR'),
                                'python', 'data', 'kernel_database.fits')
        kernelfile = fits.open(filename)
        self.kernelbase = kernelfile[0].data
        kernelfile.close()
        self.nkernelbase = self.kernelbase.shape[1]

        filename = os.path.join(os.getenv('RECONSTRUCTION_DIR'),
                                'python', 'data', 'kernel_radial_database.fits')
        kernelradial = fits.open(filename)
        self.kernel_radial = kernelradial[0].data
        kernelradial.close()

    def _kernel(self, fwhm):
        """acquire kernel values corresponding to a given fwhm

        Parameters:
        ----------

        fwhm: float, np.float32
            the fwhm of kernel(range:0.5-3.0)

        Returns:
        -------

        kernel: ndarray of np.float32
            kernel value [nkernel,nkernel]

"""
        index = int(fwhm * 1000 - 500)
        index2 = fwhm * 1000 - 500 - index
        kernel = self.kernelbase[index] * (1 - index2) + self.kernelbase[index + 1] * index2
        index_p = int((self.nkernel + self.nkernelbase) / 2)
        index_m = int((self.nkernel - self.nkernelbase) / 2)
        kernelvalue = np.zeros([self.nkernel, self.nkernel])
        kernelvalue[index_m:index_p, index_m:index_p] = kernel.copy()
        return kernelvalue

    def set_flux_psf(self, xcen=0., ycen=0., alpha=1, noise=None):
        """Set the fiber fluxes to a PSF corresponding to the kernel

        Parameters:
        ----------

        xcen : float, np.float32
            X center of PSF desired

        ycen : float, np.float32
            Y center of PSF desired

        alpha : deviation ratio of actual PSF with regard to assumed PSF (default:1)

        noise : scale factor of simulated flux to generate noise

        Notes:
        -----

        Requires set_kernel() to have already been called to set the
        flux for each exposure and wavelength.

        Only uses wavelengths specified by the object's waveindex attribute or all the wavelengths if waveindex not given.

        Sets attributes:

         .flux_psf - flux in each fiber for simulation [nExp * nfiber, nWave]
         .flux_psf_ivar - inverse variance of flux in each fiber for simulation
                      [nExp * nfiber, nWave]
         .imagevalue - simulated image [nExp, nWave, nkernel, nkernel]

"""
        if (xcen is None):
            xcen = 0.
        if (ycen is None):
            ycen = 0.
        if (noise is None):
            noise = 0

        self.flux_psf = np.zeros([self.xpos.shape[0], self.nWave])
        self.flux_psf_ivar = np.ones([self.xpos.shape[0], self.nWave],
                                     dtype=np.float32)
        self.flux_psf_mask = np.zeros([self.xpos.shape[0], self.nWave])
        imagevalue = []
        for iWave in np.arange(self.nWave):
            for iExp in np.arange(self.nExp):
                xsample = self.xpos[iExp * self.nfiber:(iExp + 1) * self.nfiber, iWave]
                ysample = self.ypos[iExp * self.nfiber:(iExp + 1) * self.nfiber, iWave]
                dx = xsample - xcen
                dy = ysample - ycen
                dd = np.zeros((len(xsample), 2))
                dd[:, 0] = dx.flatten()
                dd[:, 1] = dy.flatten()
                if (iExp < self.nExp):
                    kernelvalue = self._kernel(self.fwhm[iWave, iExp] * alpha)
                else:
                    kernelvalue = self._kernel(np.mean(self.fwhm[iWave, :]) * alpha)

                self.flux_psf[iExp * self.nfiber:(iExp + 1) * self.nfiber, iWave] = interpolate.interpn(
                    (self.xkernel, self.ykernel), kernelvalue, dd, method='linear',
                    bounds_error=False, fill_value=0.) * np.pi / self.dkernel ** 2
                if (self.nWave <= 4):
                    imagevalue.append(kernelvalue)
        if (self.nWave <= 4):
            self.imagevalue = np.array(imagevalue).reshape(self.nExp, self.nWave, self.nkernel, self.nkernel)

        if (noise != 0):
            self.flux_psf0 = self.flux_psf.copy()
            for i in range(len(self.flux_psf)):
                self.flux_psf[i] = np.random.poisson(self.flux_psf[i] * 10 ** noise) / 10 ** noise

    def set_flux_rss(self):
        """Set the flux to the RSS input values

        Notes:
        -----

        Only uses wavelengths specified by the object's waveindex attribute or all the wavelengths if waveindex not given.

        Sets attributes:

         .flux - flux in each fiber [nExp * nfiber, nWave]
         .flux_ivar - inverse variance of flux in each fiber
                      [nExp * nfiber, nWave]
         .flux_mask - mask of each fiber [nExp * nfiber, nWave]
"""
        self.flux = self.rss.data['FLUX'].data
        self.flux_ivar = self.rss.data['IVAR'].data
        self.flux_mask = self.rss.data['MASK'].data
        if (self.waveindex is not None):
            self.flux = self.flux[:, self.waveindex]
            self.flux_ivar = self.flux_ivar[:, self.waveindex]
            self.flux_mask = self.flux_mask[:, self.waveindex]

    def create_weights(self, xsample=None, ysample=None, ivar=None, waveindex=None):
        """Calculate weights for nearest fiber

        Parameters:
        ----------

        xsample : ndarray of np.float32
            X position of samples

        ysample : ndarray of np.float32
            Y position of samples

        ivar : ndarray of np.float32
            inverse variance of samples

        Returns:
        -------

        wwT : ndarray of np.float32
            normalized weights [nside * nside, nExp * nfiber]

        Notes:
        -----

        This version just sets the weight to unity for the nearest fiber.
"""
        iok = np.where(ivar > 0.)
        w = np.zeros([len(xsample), self.nside * self.nside], dtype=np.float32)
        for i in np.arange(self.nside):
            for j in np.arange(self.nside):
                dx = xsample - self.x2i[i, j]
                dy = ysample - self.y2i[i, j]
                r = np.sqrt(dx ** 2 + dy ** 2)
                iclosest = r[iok].argmin()
                w[iclosest, i * self.nside + j] = 1.
        wwT = self.normalize_weights(w)
        return (wwT)

    def normalize_weights(self, w):
        """Normalize weights

        Parameters:
        ----------

        w : ndarray of np.float32
            weights [nExp * nfiber, nside * nside]

        Returns:
        -------

        wwT : ndarray of np.float32
            normalized weights [nside * nside, nExp * nfiber]

        Notes:
        -----

        Normalizes the contributions of each pixel over all fiber-exposures.
        If the pixel has no contributions, sets weight to zero.
"""
        wsum = w.sum(axis=0)

        ww = np.zeros(w.shape, dtype=np.float32)
        for i in np.arange(self.nimage):
            if wsum[i] == 0:
                ww[:, i] = 0
            else:
                ww[:, i] = w[:, i] / wsum[i]
        wwT = ww.T
        return (wwT)

    def set_cube(self, psf_slices=True, **kwargs):
        """Set cube for each wavelength

        Notes:
        -----

        Sets the attributes:

          .cube - cube of RSS as ndarray of np.float32
                  [nside, nside, nWave]
          .cube_ivar - inverse variance of cube as ndarray of np.float32
                       [nside, nside, nWave]
          .cube_corr - correlation matrix

          .cube_psf - cube of simulation as ndarray of np.float32
                  [nside, nside, nWave]
          .cube_ivar - inverse variance of cube of simulation as ndarray of np.float32
                       [nside, nside, nWave]
"""
        self.psf_slices = psf_slices
        self.cube_psf, self.cube_psf_ivar, self.cube_psf_corr, self.cube_psf_mask = self.calculate_cube(self.flux_psf,
                                                                                                        self.flux_psf_ivar,
                                                                                                        self.flux_psf_mask,
                                                                                                        keyword='simulation',
                                                                                                        psf_slices=psf_slices,
                                                                                                        **kwargs)
        self.cube, self.cube_ivar, self.cube_corr, self.cube_mask = self.calculate_cube(self.flux, self.flux_ivar,
                                                                                        self.flux_mask, keyword='real',
                                                                                        psf_slices=False, **kwargs)
        return

    def calculate_cube(self, flux, flux_ivar, flux_mask, keyword, psf_slices=False, **kwargs):
        """calculate cube and cube inverse variance for given flux

        Parameters:
        ----------

        flux: ndarray of np.float32
            flux of each fiber [nExp * nfiber, nWave]

        flux_ivar: ndarray of np.float32
            flux inverse variance of each fiber [nExp * nfiber, nWave]

        flux_mask: ndarray of np.int32
            mask of each fiber [nExp * nfiber, nWave]

        Return:
        ------

        cube : ndarray of np.float32 [nside, nside, nWave]
            reconstruction result
        cube_ivar : ndarray of np.float32 [nside, nside, nWave]
            inverse variance of reconstruction result
        cube_corr : list of correlation matrix in sparse array format.
            If waveindex is None, return the correlation matrix at SDSS g,r,i,z broadband effective wavelengths;
            else, return the correlation matrix first four wavelength indexes
        cube_mask : ndarray of np.int32 [nside, nside, nWave]
            mask of reconstruction pixels

        Notes:
        -----
        For the actual flux, nWave is the number of input wavelength slices.
        For the simulation case, if  waveindex is None, return the result at SDSS g,r,i,z broadband effective wavelengths.
        else if the input is more than 4 wavelength slices, then choose the first four indexes.

"""
        if psf_slices:
            if self.waveindex is None:  ## if calculate slices for PSF, and set as spectrum, choose 4 bands
                waveindex = self.bbindex
                nWave = len(waveindex)
            else:  # else,choose 4 at most
                nWave = min(self.nWave, 4)
                waveindex = self.waveindex[0:nWave]
        else:
            if self.waveindex is None:  # if real,choose spectrum
                waveindex = np.arange(self.nWave)
                nWave = len(waveindex)
            else:
                nWave = self.nWave
                waveindex = self.waveindex
        cube = np.zeros([nWave, self.nside, self.nside],
                        dtype=np.float32)

        cube_ivar = np.zeros([nWave, self.nside, self.nside],
                             dtype=np.float32)
        i = 0
        cube_corr = []
        self.slice_fail = []
        cube_mask = np.zeros([nWave, self.nside, self.nside],
                             dtype=np.int32)
        if nWave <= 3 and keyword == 'simulation':
            self.weights_psf = np.zeros([nWave, self.nside * self.nside,
                                         self.nExp * self.nfiber])
        if nWave <= 3 and keyword == 'real':
            self.weights = np.zeros([nWave, self.nside * self.nside,
                                     self.nExp * self.nfiber])
        for iWave in np.arange(nWave):
            if self.waveindex is None and psf_slices is True:
                iWave = self.bbindex[iWave]
            try:
                w0, weights = self.create_weights(xsample=self.xpos[0:self.nExp * self.nfiber, iWave],
                                                  ysample=self.ypos[0:self.nExp * self.nfiber, iWave],
                                                  ivar=flux_ivar[:, iWave],
                                                  waveindex=iWave, **kwargs)
            except:
                print('failing to converge', iWave)
                self.slice_fail.append(iWave)
            self.w0 = w0
            if nWave <= 3 and keyword == 'simulation':
                self.weights_psf[iWave, :, :] = weights
            elif keyword == 'real' and nWave <= 3:
                self.weights[iWave, :, :] = weights
            fcube = ((weights.dot(flux[:, iWave])).reshape(self.nside,
                                                           self.nside) *
                     self.conversion)
            cube[i, :, :] = fcube

            # covariance
            covar = self.covar(iWave, flux_ivar, weights) * self.conversion ** 2
            var = np.diagonal(covar)
            igt0 = np.where(var > 0)[0]
            ivar = np.zeros(self.nside * self.nside, dtype=np.float32)
            ivar[igt0] = 1. / var[igt0]
            cube_ivar[i, :, :] = ivar.reshape([self.nside,
                                               self.nside])

            # correlation matrix only available for up to four wavelength slices
            if self.waveindex is None:
                if iWave in self.bbindex:
                    corr = covar / np.outer(np.sqrt(var), np.sqrt(var))
                    corr[np.where(covar == 0)] = 0
                    cube_corr.append(sparse.csr_matrix(corr))
            elif i < 4:
                corr = covar / np.outer(np.sqrt(var), np.sqrt(var))
                corr[np.where(covar == 0)] = 0
                cube_corr.append(sparse.csr_matrix(corr))

                # mask
            cube_mask[i, :, :] = self.mask(iWave, flux_ivar, flux_mask, w0, weights).reshape([self.nside, self.nside])

            i = i + 1

        return (cube, cube_ivar, cube_corr, cube_mask)

    def covar(self, iWave=None, flux_ivar=None, weights=None):
        """Return cube covariance matrix for a wavelength

        Parameters:
        ----------

        iWave : int, np.int32
            index of wavelength to calculate for

        flux_ivar: ndarray of np.float32
            flux inverse variance of each fiber [nExp * nfiber]

        weights: ndarray of np.float32
            weights matrix between pixels and fibers [nside * nside, nExp * nfiber]

        Returns:
        -------

        covar : ndarray of np.float32 [nside * nside, nside * nside]
"""
        iok = np.where(flux_ivar[:, iWave] > 0)[0]
        wwT = (weights[:, :])[:, iok]
        covar = wwT.dot(np.diag(1 / (flux_ivar[iok, iWave]))).dot(wwT.T)
        return (covar)

    def mask(self, iWave=None, flux_ivar=None, flux_mask=None, w0=None, weights=None):
        """Return mask matrix for a typical wavelength given the weights matrix

        Parameters:
        ----------

        iWave : int, np.int32
            index of wavelength to calculate for

        flux_ivar: ndarray of np.float32
            flux inverse variance of each fiber [nExp * nfiber]

        flux_mask : ndarray of np.int32
            mask of each fiber [nExp * nfiber]

        weights: ndarray of np.float32
            weights matrix between pixels and fibers [nside * nside, nExp * nfiber]

        Returns:
        -------

        maskimg : ndarray of np.int32 [nside * nside, nside * nside]

"""
        flagdeadfiber = sdss_flagval('MANGA_DRP3PIXMASK', 'DEADFIBER')
        flaglocov = sdss_flagval('MANGA_DRP3PIXMASK', 'LOWCOV')
        flagnocov = sdss_flagval('MANGA_DRP3PIXMASK', 'NOCOV')
        flagnouse = sdss_flagval('MANGA_DRP3PIXMASK', 'DONOTUSE')
        flag3dreject = sdss_flagval('MANGA_DRP2PIXMASK', '3DREJECT')
        mask_dead, mask_lowcov, mask_nocov, mask_dnu = [np.zeros(self.nside ** 2) for i in range(4)]

        index_nocov = np.where(w0.sum(axis=0) == 0)[0]
        mask_nocov[index_nocov] = flagnocov
        mask_dnu[index_nocov] = flagnouse

        #         index_goodfibers=np.where(flux_ivar[:,iWave])
        #         ngood=w0[index_goodfibers].sum(axis=0)
        #         mask_lowcov[np.where(ngood<0.5*ngood.max())]=flaglocov
        #         mask_dnu[np.where(ngood<0.5*ngood.max())]=flagnouse

        cov = np.diag(weights.dot(weights.T))
        cov = cov / np.median(cov[np.where(cov)])
        indices = np.logical_or(cov == 0, cov > 2)
        mask_lowcov[indices] = flaglocov
        mask_dnu[indices] = flagnouse

        #         index_goodfibers=np.where(flux_ivar[:,iWave])
        #         ngood = ((weights[index_goodfibers]!=0).sum(axis=0))
        #         mask_lowcov[np.where(ngood<0.30*ngood.max())]=flaglocov
        #         mask_dnu[np.where(ngood<0.30*ngood.max())]=flagnouse

        index_deadfibers = np.where(np.bitwise_and(np.uint32(flagdeadfiber), np.uint32(flux_mask[:, iWave])))
        mask_dead = ((w0[index_deadfibers] != 0).sum(axis=0) != 0) * flagdeadfiber

        maskimg = np.uint32(mask_nocov)
        maskimg = np.bitwise_or(maskimg, np.uint32(mask_lowcov))
        maskimg = np.bitwise_or(maskimg, np.uint32(mask_dead))
        maskimg = np.bitwise_or(maskimg, np.uint32(mask_dnu))

        return maskimg

    def plot_slice(self, iWave=0, keyword=None, vmax=None, vmin=0.):
        """Plot a slice of the cube

        Parameters:
        ----------

        iWave : default=0,int, np.int32
            index of wavelength to plot, order in waveindex, not the value.

        keyword : 'simulation' or else
            if keyword == 'simulation': plot reconstruction from simulated flux
            else plot reconstruction from real flux

        vmax,vmin: maximum and minimum of plot
            can be set as desired. defaulted vmax is set as maximum of flux, vmin is set as 0.


"""
        if keyword == 'simulation':
            target = self.cube_psf[iWave, :, :]
        else:
            target = self.cube[iWave, :, :]
        map = target
        if (vmax is None):
            vmax = map.max() * 1.02
        extent = (self.xmin, self.xmax, self.ymin, self.ymax)
        plt.figure(figsize=(6.5, 5.))
        font = {'family': 'sans-serif',
                'size': 15}
        plt.rc('font', **font)
        plt.imshow(target, extent=extent, vmin=vmin,
                   vmax=vmax, cmap=cm.gray_r, origin='lower')
        plt.xlabel('X (arcsec)')
        plt.ylabel('Y (arcsec)')
        plt.title('reconstruction of ' + keyword + ' slice')
        plt.colorbar(label='flux')

    def set_band(self):
        """ set average result for each band simulation and RSS and its FWHM

        Notes:
        -----

        Only uses the full range wavelengths will give the band average

        Sets attributes:

         .GPSF/RPSF/IPSF/ZPSF - ndarray of np.float32 [nside, nside] simulation image for each broadband
         .GIMG/RIMG/IIMG/ZIMG - ndarray of np.float32 [nside, nside] real image for each broadband
         .GFWHM/RFWHM/IFWHM/ZFWHM - float, np.float32,FWHM for each simulation image

"""

        if self.psf_slices:
            self.GPSF = self.cube_psf[0, :, :]
            self.RPSF = self.cube_psf[1, :, :]
            self.IPSF = self.cube_psf[2, :, :]
            self.ZPSF = self.cube_psf[3, :, :]
        else:
            self.GPSF = self.PSFaverage('g', self.wave, self.cube_psf)
            self.RPSF = self.PSFaverage('r', self.wave, self.cube_psf)
            self.IPSF = self.PSFaverage('i', self.wave, self.cube_psf)
            self.ZPSF = self.PSFaverage('z', self.wave, self.cube_psf)
        self.GIMG = self.PSFaverage('g', self.wave, self.cube)
        self.RIMG = self.PSFaverage('r', self.wave, self.cube)
        self.IIMG = self.PSFaverage('i', self.wave, self.cube)
        self.ZIMG = self.PSFaverage('z', self.wave, self.cube)
        filename = os.path.join(os.getenv('RECONSTRUCTION_DIR'),
                                'python', 'data', 'kernel_database.fits')
        kernelfile = fits.open(filename)
        self.GFWHM = kernelfile[1].data[self.chi2_index(self.GPSF)[0]]
        self.RFWHM = kernelfile[1].data[self.chi2_index(self.RPSF)[0]]
        self.IFWHM = kernelfile[1].data[self.chi2_index(self.IPSF)[0]]
        self.ZFWHM = kernelfile[1].data[self.chi2_index(self.ZPSF)[0]]
        kernelfile.close()

    def chi2_index(self, cube, xcen=None, ycen=None):
        """
        Find the index of radial fitting
    """
        if (xcen is None):
            xcen = 0.
        if (ycen is None):
            ycen = 0.
        fit_model = np.zeros([self.kernelbase.shape[0], self.nside, self.nside])
        kernelvalue2 = np.zeros([3 * self.nkernel, 3 * self.nkernel])

        index_p = int((self.nkernel + self.nkernelbase) / 2)
        index_m = int((self.nkernel - self.nkernelbase) / 2)
        start = int((self.nkernel - 1) / 2 - (self.nside - 1) / 2 * self.dimage / self.dkernel)
        gap = int(self.dimage / self.dkernel)
        for i in range(self.kernelbase.shape[0]):
            kernelvalue = np.zeros([self.nkernel, self.nkernel])
            kernelvalue[index_m:index_p, index_m:index_p] = self.kernelbase[i]
            if xcen or ycen:
                kernelvalue2[self.nkernel:2 * self.nkernel, self.nkernel:2 * self.nkernel] = kernelvalue
                dx = int(ycen / self.dkernel)
                dy = int(xcen / self.dkernel)
                kernelvalue = kernelvalue2[self.nkernel - dx:2 * self.nkernel - dx,
                              self.nkernel - dy:2 * self.nkernel - dy]
            fit_model[i] = kernelvalue[start::gap, start::gap]

        A = np.zeros(self.kernelbase.shape[0])
        chi2 = np.zeros_like(A)
        for i in range(self.kernelbase.shape[0]):
            A[i] = (fit_model[i] * cube).sum() / (fit_model[i] ** 2).sum()
            chi2[i] = ((A[i] * fit_model[i] - cube) ** 2).sum()

        ind = np.argmin(chi2)
        return ind, A[ind]

    def PSFaverage(self, color=None, wave=None, PSF=None):
        """ calculate FWHM for given image

        Parameters:
        ----------

        color : str, the color of band, can choose 'g'/'r'/'i'/'z'

        wave : ndarray of np.float32
            the spectrum of flux

        PSF : ndarray of np.float32 [nside,nside,nWave]
            the spectrum of cube

        Returns:
        -------

        PSF_ave : ndarray of np.float32 [nside,nside]
            the average of cube for given band
"""
        filterfile = os.path.join(os.getenv('RECONSTRUCTION_DIR'),
                                  'python', 'data', color + '_filter.dat')
        band0 = np.loadtxt(filterfile)
        band1 = np.arange(3400, band0[0, 0], 25)
        band2 = np.arange(band0[-1, 0], 11000, 25)
        weight1 = np.zeros(band1.shape)
        weight2 = np.zeros(band2.shape)
        band = np.concatenate((np.concatenate((band1, band0[:, 0]), axis=0), band2), axis=0)
        weight = np.concatenate((np.concatenate((weight1, band0[:, 1]), axis=0), weight2), axis=0)
        fun_band = interpolate.interp1d(band, weight)
        band_value = fun_band(wave)

        n = PSF.shape[1]
        nWave = len(wave)
        PSF_ave = (np.matlib.repmat(band_value, n ** 2, 1).T * (PSF.reshape(nWave, n ** 2))).reshape(nWave, n, n).sum(
            axis=0) / band_value.sum()
        return PSF_ave


class ReconstructShepard(Reconstruct):
    """Reconstruction of cubes from Shepards method

    Attributes:
    ----------

    plate : int, np.int32
        plate number

    ifu : int, np.int32
        IFU number

    nfiber : int
        number of fibers

    release : str
        data release (default 'MPL-5')

    rss : RSS object
        Marvin RSS output

    waveindex : int, np.int32
        indices of wavelengths to reconstruct (default None)

    Methods:
    -------

    set_rss() : Acquire the RSS data

    set_image_grid(dimage) : Set up the spatial grid for the cube

    set_kernel(fwhm) : Create the kernel estimation from accessing to data base

    set_flux_rss() : Set the flux used for reconstruction from RSS

    set_flux_psf(xcen=0., ycen=0.,alpha=1,noise=None) : Set the flux used for reconstruction to a PSF

    set_weights() : Set the weights for mapping spectra to cube

    create_weights() : Calculate the weights for mapping spectra to cube

    normalize_weights(w) : Normalize the weights

    set_cube() : Calculates and sets cube for both RSS and simulation

    calculate_cube(flux,flux_ivar,flux_mask): Calculate the result for given flux, flux_ivar and flux_mask

    covar() : Calculate covariance matrix for a slice of the cube

    mask(): Calculate mask matrix for a slice of the cube

    plot_slice() : Plots a slice of the cube

    FWHM(xi=None, yi=None, PSF=None, xcen=0, ycen=0): calculate FWHM for given reconstructed image

    Notes:
    ------

    Additional attributes are set by the methods, as documented.

    Unless waveindex is set, uses all wavelengths.

    Typical usage would be (for a limited number of wavelengths):

     import reconstruction
     r = reconstruction.Reconstruct(plate=8485, ifu=3701, waveindex=[1000, 2000])
     r.set_rss() # Gets and reads in the RSS file (sets .rss attribute)
     r.set_image_grid() # Creates the output image spatial grid
     r.set_kernel() # Sets the kernel for every wavelength and exposure
     r.set_flux_rss() # Sets up to use the RSS fluxes (sets .flux, .flux_ivar)
     r.set_flux_psf(xcen=0.5, ycen=1.5, alpha=1,noise=5) # Puts fake data into PSF fluxes (sets .flux_psf, .flux_psf_ivar)
     r.set_weights() # Sets the weights (sets .weights)
     r.set_cube() # Sets the weights (sets .cube, .cube_ivar .cube_mask)
     r.plot_slice(0)
"""

    def create_weights(self, xsample=None, ysample=None,
                       ivar=None, waveindex=None, shepard_sigma=0.7):
        """Calculate weights for Shepards method

        Parameters:
        ----------

        xsample : ndarray of np.float32
            X position of samples

        ysample : ndarray of np.float32
            Y position of samples

        ivar : ndarray of np.float32
            inverse variance of samples

        shepard_sigma : float, np.float32
            sigma for Gaussian in kernel, in arcsec (default 0.7)

        Returns:
        -------

        w0 : ndarray of np.float32
            unnormalized weights without bad fibers, [nExp * nfiber,nside * nside]

        wwT : ndarray of np.float32
            normalized weights [nside * nside, nExp * nfiber]

        Notes:
        -----

        This version uses Shepards method.
"""
        nsample = len(xsample)
        dx = (np.outer(xsample, np.ones(self.nimage, dtype=np.float32)) -
              np.outer(np.ones(nsample, dtype=np.float32), self.x2i.flatten()))
        dy = (np.outer(ysample, np.ones(self.nimage)) -
              np.outer(np.ones(nsample, dtype=np.float32), self.y2i.flatten()))
        dr = np.sqrt(dx ** 2 + dy ** 2)

        w0 = np.exp(- 0.5 * dr ** 2 / shepard_sigma ** 2)
        ifit = np.where(dr > 1.6)
        w0[ifit] = 0
        w = np.transpose(np.matlib.repmat(ivar != 0, self.nside ** 2, 1)) * w0

        wwT = self.normalize_weights(w)
        return (w0, wwT)


class ReconstructG(Reconstruct):
    """Reconstruction of cubes from linear least square method

    Attributes:
    ----------

    plate : int, np.int32
        plate number

    ifu : int, np.int32
        IFU number

    nfiber : int
        number of fibers

    release : str
        data release (default 'MPL-5')

    rss : RSS object
        Marvin RSS output

    waveindex : int, np.int32
        indices of wavelengths to reconstruct (default None)

    Methods:
    -------


    set_rss() : Acquire the RSS data

    set_image_grid(dimage) : Set up the spatial grid for the cube

    set_kernel(fwhm) : Create the kernel estimation from accessing to data base

    set_flux_rss() : Set the flux used for reconstruction from RSS

    set_flux_psf(xcen=0., ycen=0.,alpha=1,noise=None) : Set the flux used for reconstruction to a PSF

    set_weights() : Set the weights for mapping spectra to cube

    create_weights() : Calculate the weights for mapping spectra to cube

    normalize_weights(w) : Normalize the weights

    set_cube() : Calculates and sets cube for both RSS and simulation

    calculate_cube(flux,flux_ivar,flux_mask): Calculate the result for given flux, flux_ivar and flux_mask

    covar() : Calculate covariance matrix for a slice of the cube

    mask(): Calculate mask matrix for a slice of the cube

    plot_slice() : Plots a slice of the cube

    FWHM(xi=None, yi=None, PSF=None, xcen=0, ycen=0): calculate FWHM for given reconstructed image

    Notes:
    ------

    Additional attributes are set by the methods, as documented.

    Unless waveindex is set, uses all wavelengths.

    Typical usage would be (for a limited number of wavelengths):

     import reconstruction
     r = reconstruction.Reconstruct(plate=8485, ifu=3701, waveindex=[1000, 2000])
     r.set_rss() # Gets and reads in the RSS file (sets .rss attribute)
     r.set_image_grid() # Creates the output image spatial grid
     r.set_kernel() # Sets the kernel for every wavelength and exposure
     r.set_flux_rss() # Sets up to use the RSS fluxes (sets .flux, .flux_ivar)
     r.set_flux_psf(xcen=0.5, ycen=1.5, alpha=1,noise=5) # Puts fake data into PSF fluxes (sets .flux_psf, .flux_psf_ivar)
     r.set_weights() # Sets the weights (sets .weights)
     r.set_cube() # Sets the weights (sets .cube, .cube_ivar .cube_mask)
     r.plot_slice(0)

"""

    def set_Amatrix(self, xsample=None, ysample=None, ivar=None, waveindex=None, ratio=30, beta=1):
        """Calculate kernel matrix for linear least square method

        Parameters:
        ----------

        xsample : ndarray of np.float32
            X position of samples

        ysample : ndarray of np.float32
            Y position of samples

        kernel : float, np.float32
            kernel at each and exposure [nExp, nkernel, nkernel]

        lam : regularization factor (default 1E-4)

        ratio: criterion to select pixels (default 30)

        Returns:
        -------

        ifit: indices of pixels selected

        A : ndarray of np.float32
            kernel matrix [nExp * nfiber, nfit]

        Notes:
        -----

        indices will be used to recover A matrix back to regular grid
"""
        nsample = len(xsample)
        dx = np.outer(xsample, np.ones(self.nimage)) - np.outer(np.ones(nsample), self.x2i.flatten())
        dy = np.outer(ysample, np.ones(self.nimage)) - np.outer(np.ones(nsample), self.y2i.flatten())
        dr = np.sqrt(dx ** 2 + dy ** 2)
        ifit = np.where(dr.min(axis=0) <= 1.6)[0]

        dr = dr[:, ifit]

        dr = dr.flatten()
        dfwhm = (np.matlib.repmat(self.fwhm[waveindex], self.nfiber * len(ifit), 1).flatten('F'))

        radius_lim = 4
        indices = np.where(dr.flatten() < radius_lim)[0]

        dd = np.zeros([len(indices), 2])
        dd[:, 0] = dfwhm.flatten()[indices]
        dd[:, 1] = dr.flatten()[indices]

        ifwhm = np.arange(0.5, 2.5, 0.001)
        fwhmmin = int(self.fwhm.min() * 1000) - 500
        fwhmmax = int(self.fwhm.max() * 1000) - 500
        ifwhm = ifwhm[max(fwhmmin - 3, 0):min(fwhmmax + 3, 2000)]

        ir = np.arange(0, 5.5, 0.05)

        Afull = interpolate.interpn((ifwhm, ir), self.kernel_radial[max(fwhmmin - 3, 0):min(fwhmmax + 3, 2000), :], dd,
                                    method='linear', bounds_error=False, fill_value=0.) * (
                                                                                          self.dimage / self.dkernel) ** 2
        Afull2 = np.zeros(len(dr.flatten()))
        Afull2[indices] = Afull
        A = Afull2.reshape(self.nExp * self.nfiber, len(ifit))

        return (ifit, A)

    def create_weights(self, xsample=None, ysample=None, ivar=None, waveindex=None, lam=0, ratio=30, beta=1):
        """Calculate weights for linear least square method

        Parameters:
        ----------

        xsample : ndarray of np.float32
            X position of samples

        ysample : ndarray of np.float32
            Y position of samples

        ivar : ndarray of np.float32
            inverse variance of samples

        kernel : float, np.float32
            kernel at each and exposure [nExp, nkernel, nkernel]

        lam : regularization factor (default 1E-4)

        ratio: criterion to select pixels (default 30)

        Returns:
        -------
        w0 : ndarray of np.float32
            unnormalized weights without bad fibers, [nExp * nfiber,nside * nside]

        wwT : ndarray of np.float32
            normalized weights [nside * nside, nExp * nfiber]

        Notes:
        -----

"""
        self.ifit, A = self.set_Amatrix(xsample, ysample, ivar, waveindex, ratio, beta)
        self.nfit = len(A[0])
        ivar = (ivar != 0)
        [U, D, VT] = np.linalg.svd(np.dot(np.diag(np.sqrt(ivar)), A), full_matrices=False)
        Dinv = 1 / D

        self.lam = lam
        for i in range(len(D)):
            if D[i] < 1E-6:
                Dinv[i] = 0
        filt = 1 / (1 + lam ** 2 * Dinv ** 2)

        A_plus = np.dot(np.dot(VT.T, np.dot(np.diag(filt), np.diag(Dinv))), U.T)

        Q = (np.dot(np.dot(VT.transpose(), np.dot(np.diag(1 / filt), np.diag(D))), VT))
        sl = Q.sum(axis=1)
        Rl = (Q.T / sl.T).T
        where_are_NaNs = np.isnan(Rl)
        Rl[where_are_NaNs] = 0

        T = np.dot(np.dot(Rl, A_plus), np.diag(np.sqrt(ivar)))
        self.A_plus = self.set_reshape(np.dot(A_plus, np.diag(np.sqrt(ivar))))
        wwT = self.set_reshape(T)
        return (self.set_reshape(A.T).T, wwT)

    def set_reshape(self, inp):
        """ reshape the size of weights from selected pixels to a regular grid

        Parameters:
        ----------
        inp : ndarray of np.float32
            input array, [nfit, size]

        Return:
        --------
        output : ndarray of np.float32
            output array, [nside * nside, size]

"""
        output = np.zeros([self.nside ** 2, inp.shape[1]])
        output[self.ifit] = inp
        return output


def set_base(plate=None, ifu=None, release='MPL-5', waveindex=None, dimage=0.75, dkernel=0.05):
    """ set the grid and fiber fluxes

        Parameters:
        ----------

        plate : int, np.int32
            plate number

        ifu : int, np.int32
            IFU number

        release : str
            data release (default 'MPL-5')

        waveindex : int, np.int32
            indices of wavelengths to reconstruct (default None)

        Return:
        --------
        base: object that includes information of grid and fiber fluxes

"""
    base = Reconstruct(plate=plate, ifu=ifu, release=release, waveindex=waveindex)
    base.set_rss()
    base.set_image_grid(dimage=dimage)
    base.set_kernel(dkernel=dkernel)
    base.set_flux_rss()
    base.set_flux_psf()
    return base


def set_G(plate=None, ifu=None, release='MPL-5', waveindex=None, dimage=0.75, lam=0, alpha=1, beta=1, xcen=None,
          ycen=None, noise=None, psf_slices=True):
    """ Linear least square method for the reconstruction

        Parameters:
        ----------

        plate : int, np.int32
            plate number

        ifu : int, np.int32
            IFU number

        release : str
            data release (default 'MPL-5')

        waveindex : int, np.int32
            indices of wavelengths to reconstruct (default None)

        dimage: default=0.75, np.float32
            pixel size of reconstruction grid

        lam: default=1E-3,np.float32
            regularization factor

        alpha: default=1,np.float32
            scale factor to change the FWHM of actual seeing

        beta: default=1,np.float32
            scale factor to change the FWHM of assumed seeing

        xcen: default=0,np.float32
            x coordinate of the location of center for the simulation case

        ycen: default=0,np.float32
            y coordinate of the location of center for the simulation case

        noise: default=0,np.float32
             scale factor of simulated flux to generate noise

        Return:
        --------
        base: object that includes information of grid and fiber fluxes and reconstruction image

"""
    base = ReconstructG(plate=plate, ifu=ifu, release=release, waveindex=waveindex)
    base.set_rss()
    base.set_image_grid(dimage=dimage)
    base.set_kernel()
    base.set_flux_rss()
    base.set_flux_psf(alpha=alpha, xcen=xcen, ycen=ycen, noise=noise)
    start_time = time.time()
    base.set_cube(psf_slices=psf_slices, beta=beta, lam=lam)
    stop_time = time.time()
    print("G‘s calculation time = %.2f" % (stop_time - start_time))
    if (len(base.wave) == base.rss.data['FLUX'].data.shape[1]):
        base.set_band()
    return base


def set_Shepard(plate=None, ifu=None, release='MPL-5', waveindex=None, dimage=0.75, alpha=1, beta=1, xcen=None,
                ycen=None, noise=None, psf_slices=True):
    """ Shepard's method for the reconstruction

        Parameters:
        ----------

        plate : int, np.int32
            plate number

        ifu : int, np.int32
            IFU number

        release : str
            data release (default 'MPL-5')

        waveindex : int, np.int32
            indices of wavelengths to reconstruct (default None)

        alpha: default=1,np.float32
            scale factor to change the FWHM of actual seeing

        beta: default=1,np.float32
            scale factor to change the FWHM of assumed seeing

        xcen: default=0,np.float32
            x coordinate of the location of center for the simulation case

        ycen: default=0,np.float32
            y coordinate of the location of center for the simulation case

        noise: default=0,np.float32
             scale factor of simulated flux to generate noise

        Return:
        --------
        base: object that includes information of grid and fiber fluxes

"""
    base = ReconstructShepard(plate=plate, ifu=ifu, release=release, waveindex=waveindex)
    base.set_rss()
    base.set_image_grid(dimage=dimage)
    base.set_kernel()
    base.set_flux_rss()
    base.set_flux_psf(alpha=alpha, xcen=xcen, ycen=ycen, noise=noise)
    start_time = time.time()
    base.set_cube(psf_slices=psf_slices)
    stop_time = time.time()
    print("calculation time = %.2f" % (stop_time - start_time))
    if (len(base.wave) == base.rss.data['FLUX'].data.shape[1]):
        base.set_band()
    return base


def write(datafile, filename):
    """ write the information for a particular plate-ifu to a FITS file as MaNGA cube form

        Parameters:
        ----------
        datafile: object that includes all the information for MaNGA reconstruction

        filename: str, the name of fits file


        Return:
        --------
        hdu: FITS file that includes all the information as MaNGA cube form

"""

    def insert_cardlist(hdu=None, insertpoint=None, cardlist=None, after=False):
        """ insert a cardlist into the header of a FITS hdu

        Parameters:
        ----------
        hdu: FITS hdu

        insertpoint: The index into the list of header keywords before which the new keyword should be inserted, or the name of a keyword before which the new keyword should be inserted.
            Can also accept a (keyword, index) tuple for inserting around duplicate keywords.

        cardlist: list
            list of header cards to be inserted. Header cards will be inserted as the order of the list

        after: bool, optional

            If set to True, insert after the specified index or keyword, rather than before it. Defaults to False.

        Return:
        --------
        hdu: fits file that have the cards inserted

    """
        for i in range(len(cardlist)):
            if after:
                hdu.header.insert(insertpoint, cardlist[i], after=after)
                insertpoint = cardlist[i].keyword
            else:
                hdu.header.insert(insertpoint, cardlist[i], after=after)
        return hdu

    def set_cardlist(hdu=None, keyword_list=None):
        """ Extract header card list from a FITS hdu

        Parameters:
        ----------
        hdu: FITS hdu

        keyword_list: list
            keywords to be extracted, including value and comments

        Return:
        --------
        cardlist: list,cards of FITS headers

    """
        cardlist = []
        for index, keyword in enumerate(keyword_list):
            cardlist.append(fits.Card(keyword, hdu.header[keyword], hdu.header.comments[keyword]))
        return cardlist

    def table_correlation(correlation=None, thresh=1E-12):
        """ create a BinTableHDU for the correlation in sparse matrix form

        Parameters:
        ----------
        correlation: ndarray,float32
            correlation matrix, [nside*nside,nside*nside]

        thresh: float32
            threshold for the correlation entries to be stored.

        Return:
        --------
        hdu: BinTableHDU that includes the information of the correlation matrix

        Note:
        --------
        Fiven columns of the table are value, the location (C1,c2) of the first point in the grid, the location (C1,c2) of the second point in the grid.

    """
        nside = int(np.sqrt(correlation.shape[0]))
        index_G = np.where(np.abs(correlation) > thresh)
        corr_G = correlation[index_G]
        triangle = np.where(index_G[1] >= index_G[0])[0]
        index_G = np.array([index_G[0][triangle], index_G[1][triangle]])
        corr_G = corr_G[triangle]
        i_c1, i_c2, j_c1, j_c2 = [[] for i in range(4)]
        for i in range(len(corr_G)):
            i_c2.append(index_G[0, i] // nside)
            i_c1.append(index_G[0, i] % nside)
            j_c2.append(index_G[1, i] // nside)
            j_c1.append(index_G[1, i] % nside)
        i1 = fits.Column(name='INDXI_C1', array=np.array(i_c1), format='J')
        i2 = fits.Column(name='INDXI_C2', array=np.array(i_c2), format='J')
        j1 = fits.Column(name='INDXJ_C1', array=np.array(j_c1), format='J')
        j2 = fits.Column(name='INDXJ_C2', array=np.array(j_c2), format='J')
        value = fits.Column(name='RHOIJ', array=np.array(corr_G), format='D')
        hdu = fits.BinTableHDU.from_columns([i1, i2, j1, j2, value])
        return hdu

    # Headers
    card_GFWHM = fits.Card('GFWHM', datafile.GFWHM, 'Reconstructed FWHM in g-band (arcsec)')
    card_RFWHM = fits.Card('RFWHM', datafile.RFWHM, 'Reconstructed FWHM in r-band (arcsec)')
    card_IFWHM = fits.Card('IFWHM', datafile.IFWHM, 'Reconstructed FWHM in i-band (arcsec)')
    card_ZFWHM = fits.Card('ZFWHM', datafile.ZFWHM, 'Reconstructed FWHM in z-band (arcsec)')
    card_FWHM_list = [card_GFWHM, card_RFWHM, card_IFWHM, card_ZFWHM]

    card_BSCALE = fits.Card('BSCALE', 1.00000, 'Intensity unit scaling')
    card_BZERO = fits.Card('BZERO', 0.00000, 'Intensity zeropoint')
    card_BSCALE_2 = fits.Card('BSCALE', 1.00000, 'Flux unit scaling')
    card_BZERO_2 = fits.Card('BZERO', 0.00000, 'Flux zeropoint')

    card_WCS_1 = fits.Card('CRPIX1', (datafile.nside - 1) / 2, 'Reference pixel (1-indexed)')
    card_WCS_2 = fits.Card('CRPIX2', (datafile.nside - 1) / 2, 'Reference pixel (1-indexed)')
    card_WCS_3 = fits.Card('CRVAL1', datafile.rss.data['FLUX'].header['IFURA'])
    card_WCS_4 = fits.Card('CRVAL2', datafile.rss.data['FLUX'].header['IFUDEC'])
    card_WCS_5 = fits.Card('CD1_1', round(-0.5 / 3600, 9))
    card_WCS_6 = fits.Card('CD2_2', round(0.5 / 3600, 9))
    card_WCS_7 = fits.Card('CTYPE1', 'RA---TAN')
    card_WCS_8 = fits.Card('CTYPE2', 'DEC---TAN')
    card_WCS_9 = fits.Card('CUNIT1', 'deg')
    card_WCS_10 = fits.Card('CUNIT2', 'deg')
    card_WCS_list = [card_WCS_1, card_WCS_2, card_WCS_3, card_WCS_4, card_WCS_5, card_WCS_6, card_WCS_7, card_WCS_8,
                     card_WCS_9, card_WCS_10]

    # Primary
    hp = fits.PrimaryHDU(header=datafile.rss.data[0].header)
    hp.header['BUNIT'] = ('1E-17 erg/s/cm^2/Ang/spaxel', 'Specific intensity (per spaxel)')
    hp.header['MASKNAME'] = ('MANGA_DRP3PIXMASK', 'Bits in sdssMaskbits.par used by mask extension')
    hp = insert_cardlist(hdu=hp, insertpoint='EBVGAL', cardlist=card_FWHM_list, after=True)
    if 'BSCALE' not in list(hp.header.keys()):
        hp.header.insert('BUNIT', card_BSCALE, after=False)
    if 'BZERO' not in list(hp.header.keys()):
        hp.header.insert('BUNIT', card_BZERO, after=False)

    # Flux
    cubehdr = fits.ImageHDU(name='FLUX', data=datafile.cube, header=datafile.rss.data['FLUX'].header)
    cubehdr.header['BUNIT'] = ('1E-17 erg/s/cm^2/Ang/spaxel', 'Specific intensity (per spaxel)')
    cubehdr.header['MASKNAME'] = ('MANGA_DRP3PIXMASK', 'Bits in sdssMaskbits.par used by mask extension')
    cubehdr.header['HDUCLAS1'] = 'CUBE'
    cubehdr.header.rename_keyword('CTYPE1', 'CTYPE3')
    cubehdr.header.rename_keyword('CRPIX1', 'CRPIX3')
    cubehdr.header.rename_keyword('CRVAL1', 'CRVAL3')
    cubehdr.header.rename_keyword('CD1_1', 'CD3_3')
    cubehdr.header.rename_keyword('CUNIT1', 'CUNIT3')
    cubehdr = insert_cardlist(hdu=cubehdr, insertpoint='EBVGAL', cardlist=card_FWHM_list, after=True)
    cubehdr = insert_cardlist(hdu=cubehdr, insertpoint='CUNIT3', cardlist=card_WCS_list, after=True)
    if 'BSCALE' not in list(cubehdr.header.keys()):
        cubehdr.header.insert('BUNIT', card_BSCALE, after=False)
    if 'BZERO' not in list(cubehdr.header.keys()):
        cubehdr.header.insert('BUNIT', card_BZERO, after=False)
    try:
        card_flux_fail = fits.Card('FAIL_SLICE', str(datafile.slice_fail), 'slices failed to converge')
        cubehdr.header.insert('ZFWHM', card_flux_fail, after=True)
    except:
        pass

    # IVAR
    ivar_hdr = fits.ImageHDU(name='IVAR', data=datafile.cube_ivar, header=datafile.rss.data['IVAR'].header)
    ivar_hdr.header['HDUCLAS1'] = 'CUBE'

    # MASK
    mask_hdr = fits.ImageHDU(name='MASK', data=datafile.cube_mask, header=datafile.rss.data['MASK'].header)
    mask_hdr.header['HDUCLAS1'] = 'CUBE'

    # IMG & PSF for each band
    card_BUNIT = fits.Card('BUNIT', 'nanomaggies/pixel')
    loc = ['IFURA', 'IFUDEC', 'OBJRA', 'OBJDEC']
    card_loc_list = set_cardlist(cubehdr, loc) + [card_BSCALE_2, card_BZERO_2, card_BUNIT]

    GIMG_hdr = fits.ImageHDU(name='GIMG', data=datafile.GIMG)
    GIMG_hdr = insert_cardlist(hdu=GIMG_hdr, insertpoint='EXTNAME',
                               cardlist=card_WCS_list + card_loc_list + [card_GFWHM], after=False)

    RIMG_hdr = fits.ImageHDU(name='RIMG', data=datafile.RIMG)
    RIMG_hdr = insert_cardlist(hdu=RIMG_hdr, insertpoint='EXTNAME',
                               cardlist=card_WCS_list + card_loc_list + [card_RFWHM], after=False)

    IIMG_hdr = fits.ImageHDU(name='IIMG', data=datafile.IIMG)
    IIMG_hdr = insert_cardlist(hdu=IIMG_hdr, insertpoint='EXTNAME',
                               cardlist=card_WCS_list + card_loc_list + [card_IFWHM], after=False)

    ZIMG_hdr = fits.ImageHDU(name='ZIMG', data=datafile.GIMG)
    ZIMG_hdr = insert_cardlist(hdu=ZIMG_hdr, insertpoint='EXTNAME',
                               cardlist=card_WCS_list + card_loc_list + [card_ZFWHM], after=False)

    GPSF_hdr = fits.ImageHDU(name='GPSF', data=datafile.GPSF)
    GPSF_hdr = insert_cardlist(hdu=GPSF_hdr, insertpoint='EXTNAME',
                               cardlist=card_WCS_list + card_loc_list + [card_GFWHM], after=False)

    RPSF_hdr = fits.ImageHDU(name='RPSF', data=datafile.RPSF)
    RPSF_hdr = insert_cardlist(hdu=RPSF_hdr, insertpoint='EXTNAME',
                               cardlist=card_WCS_list + card_loc_list + [card_RFWHM], after=False)

    IPSF_hdr = fits.ImageHDU(name='IPSF', data=datafile.IPSF)
    IPSF_hdr = insert_cardlist(hdu=IPSF_hdr, insertpoint='EXTNAME',
                               cardlist=card_WCS_list + card_loc_list + [card_IFWHM], after=False)

    ZPSF_hdr = fits.ImageHDU(name='ZPSF', data=datafile.GPSF)
    ZPSF_hdr = insert_cardlist(hdu=ZPSF_hdr, insertpoint='EXTNAME',
                               cardlist=card_WCS_list + card_loc_list + [card_ZFWHM], after=False)

    # CORR
    CORR_hdr = []
    for i in range(4):
        corr = table_correlation(correlation=datafile.cube_corr[i].toarray(), thresh=1E-12)
        corr.header.append(fits.Card('BBWAVE', datafile.bbwave[i], 'Wavelength (Angstroms)'))
        corr.header.append(fits.Card('BBINDEX', datafile.bbindex[i], 'Slice number'))
        corr.header.append(fits.Card('COVTYPE', 'Correlation'))
        corr.header.append(fits.Card('COVSHAPE', '(%s,%s)' % (datafile.nimage, datafile.nimage)))
        CORR_hdr.append(corr)
    CORR_hdr[0].header.append(fits.Card('EXTNAME', 'GCORREL'))
    CORR_hdr[1].header.append(fits.Card('EXTNAME', 'RCORREL'))
    CORR_hdr[2].header.append(fits.Card('EXTNAME', 'ICORREL'))
    CORR_hdr[3].header.append(fits.Card('EXTNAME', 'ZCORREL'))

    if datafile.cube_psf.shape[0]>4:
        # PSF
        PSF_hdr = fits.ImageHDU(name='PSF', data=datafile.cube_psf, header=cubehdr.header)
        hdu = fits.HDUList([hp, cubehdr, PSF_hdr, ivar_hdr, mask_hdr,
                            datafile.rss.data['WAVE'], datafile.rss.data['SPECRES'], datafile.rss.data['SPECRESD'],
                            datafile.rss.data['OBSINFO'],
                            GIMG_hdr, RIMG_hdr, IIMG_hdr, ZIMG_hdr, GPSF_hdr, RPSF_hdr, IPSF_hdr, ZPSF_hdr] + CORR_hdr)
    else:
        hdu = fits.HDUList([hp, cubehdr, ivar_hdr, mask_hdr,
                            datafile.rss.data['WAVE'], datafile.rss.data['SPECRES'], datafile.rss.data['SPECRESD'],
                            datafile.rss.data['OBSINFO'],
                            GIMG_hdr, RIMG_hdr, IIMG_hdr, ZIMG_hdr, GPSF_hdr, RPSF_hdr, IPSF_hdr, ZPSF_hdr] + CORR_hdr)

    data = filename + ".fits".format(filename=filename)
    hdu.writeto(data, clobber=True, checksum=True)

    hdu.close()

    return hdu