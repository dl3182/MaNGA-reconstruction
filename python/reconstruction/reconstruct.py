import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal as sign
import scipy.interpolate as interpolate
import marvin.tools.rss as rss
from marvin import config

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

    expindex : int, np.int32
        indices of exposures to use (default None)

    Methods:
    -------

    set_rss() : Acquire the RSS data

    set_flux_rss() : Set the flux used for reconstruction from RSS

    set_flux_psf() : Set the flux used for reconstruction to a PSF

    set_image_grid() : Set up the spatial grid for the cube

    psf(fwhm, x, y) : Return values of the PSF shape

    set_kernel() : Create the kernel estimation

    normalize_weights() : Normalize the weights

    create_weights() : Calculate the weights for mapping spectra to cube

    set_weights() : Set the weights for mapping spectra to cube

    set_cube() : Calculates and sets cube for all wavelengths

    covar() : Calculates covariance matrix for a slice of the cube

    plot_slice() : Plots a slice of the cube

    Notes:
    ------

    Additional attributes are set by the methods, as documented.

    Unless waveindex or expindex is set, uses all wavelengths and all
    exposures.

    Typical usage would be (for a limited number of wavelengths):

     import reconstruction
     r = reconstruction.Reconstruct(plate=8485, ifu=3701, waveindex=[1000, 2000])
     r.set_rss() # Gets and reads in the RSS file (sets .rss attribute)
     r.set_flux_rss() # Sets up to use the RSS fluxes (sets .flux, .flux_ivar)
     r.set_image_grid() # Creates the output image spatial grid
     r.set_weights() # Sets the weights (sets .weights)
     r.set_cube() # Sets the weights (sets .cube, .cube_ivar)
     r.plot_slice(0)

    To test performance on a PSF, you would want instead of set_flux_rss():

     r.set_kernel()  # Sets the kernel
     r.set_flux_psf(xcen=0.5, ycen=1.5)  # Puts fake data into fluxes

"""

    def __init__(self, plate=None, ifu=None, release='MPL-5', waveindex=None, expindex=None):
        self.plate = plate
        self.ifu = ifu
        self.plateifu = "{plate}-{ifu}".format(plate=self.plate, ifu=self.ifu)
        self.release = release
        self.nfiber = int(self.ifu / 100)
        self.rss = None
        self.waveindex = waveindex
        if (self.waveindex is not None):
            self.waveindex = self._arrayify_int32(self.waveindex)
        self.expindex = expindex
        if (self.expindex is not None):
            self.expindex = self._arrayify_int32(self.expindex)
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
         .fwhm0 - FWHM of seeing at guider wavelength (5500 Angstroms) [nExp]
         .fwhm - FWHM at each wavelength [nWave, nExp]
"""
        self.rss = rss.RSS(plateifu=self.plateifu, release=self.release)
        self.nExp = self.rss.data[0].header['NEXP']
        if (self.expindex is not None):
            self.runExp = self.expindex
        else:
            self.runExp = np.arange(self.nExp)
        self.xpos = self.rss.data['XPOS'].data
        self.ypos = self.rss.data['YPOS'].data
        self.wave = self.rss.data['WAVE'].data
        self.nWave = self.rss.data['FLUX'].shape[1]
        # Use waveindex if that was set
        if (self.waveindex is not None):
            self.nWave = len(self.waveindex)
            self.wave = self.wave[self.waveindex]

        # Set FWHM values as a function of wavelength
        self.obsinfo = self.rss.data['OBSINFO'].data
        self.fwhm0 = (self.obsinfo.field('SEEING') *
                      self.obsinfo.field('PSFFAC'))

        lambda0 = 5500.
        self.fwhm = [[self.fwhm0[i] * math.pow(lambda0 / self.wave[j], 1. / 5.)
                      for i in range(self.fwhm0.shape[0])]
                     for j in range(self.nWave)]
        self.fwhm = np.array(self.fwhm)
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

    def psf(self, fwhm, x, y):
        """Return value(s) of PSF

        Parameters:
        ----------

        fwhm : float, np.float32
            FWHM (in same units as x and y)

        x : float, np.float32, or np.ndarray of np.float32
            x position(s) to evaluate

        y : float, np.float32, or np.ndarray of np.float32
            y position(s) to evaluate


        Returns:
        -------

        val : float, np.float32, or np.ndarray of np.float32
            PSF value at x and y positions

        Notes:
        -----

        Uses an arbitrarily chosen double Gaussian PSF.
        Normalized per unit coordinate in same units as FWHM.
"""
        sig1 = fwhm / 2.355 / 1.05  # factor 1.05 to handle second Gaussian
        sig2 = 2. * sig1
        gaus1 = (np.exp(-(x ** 2 + y ** 2) / (2 * sig1 ** 2)) /
                 (2 * np.pi * sig1 ** 2))
        gaus2 = (np.exp(-(x ** 2 + y ** 2) / (2 * sig2 ** 2)) /
                 (2 * np.pi * sig2 ** 2))
        scale21 = 1. / 9.
        gaus = (gaus1 + scale21 * gaus2) / (1. + scale21)
        gaus = gaus / gaus.sum()
        return gaus

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
         .fiber - unconvolved image of fiber [nside_kernel, nside_kernel]
         .kernelvalue - kernel at each wavelength and exposure
                   [nWave, nExp, nkernel, nkernel]
"""
        self.dkernel = dkernel
        (self.nkernel, length, self.x2k, self.y2k, self.xkernel, self.ykernel) = self._create_grid(self.dkernel)

        radius = np.sqrt(self.x2k ** 2 + self.y2k ** 2)
        fiber = np.zeros([self.nkernel, self.nkernel],
                         dtype=np.float32)
        ifiber = np.where(radius < 1.0)
        fiber[ifiber] = 1.
        self.fiber = fiber / fiber.sum()

        # Now convolve with PSF for each exposure and wavelength
        self.kernelvalue = np.zeros([self.nWave, len(self.runExp), self.nkernel, self.nkernel], dtype=np.float32)
        for iWave in np.arange(self.nWave):
            if (iWave % 1000 == 10):
                print('kernel reconstruction', iWave)
            for index, iExp in enumerate(self.runExp):
                psf = self.psf(self.fwhm[iWave, index], self.x2k, self.y2k)
                self.kernelvalue[iWave, index, :, :] = sign.fftconvolve(self.fiber, psf,
                                                                        mode='same') * self.nkernel ** 2 / self.nside ** 2
        return

    def set_flux_psf(self, xcen=0., ycen=0.):
        """Set the fiber fluxes to a PSF corresponding to the kernel

        Parameters:
        ----------

        xcen : float, np.float32
            X center of PSF desired

        ycen : float, np.float32
            Y center of PSF desired

        Notes:
        -----

        Requires set_kernel() to have already been called to set the
        appropriate kernels for each exposure and wavelength.

        Only uses wavelengths specified by the object's waveindex attribute.

        Sets attributes:

         .flux - flux in each fiber [nExp * nfiber, nWave]
         .flux_ivar - inverse variance of flux in each fiber
                      [nExp * nfiber, nWave]

"""
        if (xcen is None):
            xcen = 0.
        if (ycen is None):
            ycen = 0.

        self.flux = np.zeros([self.xpos.shape[0], self.nWave])
        self.flux_ivar = np.ones([self.nExp * self.nfiber, self.nWave],
                                 dtype=np.float32)
        for iWave in np.arange(self.nWave):
            if (self.expindex is None):
                for iExp in np.arange(self.nExp):
                    xsample = self.xpos[iExp * self.nfiber:(iExp + 1) * self.nfiber, iWave]
                    ysample = self.ypos[iExp * self.nfiber:(iExp + 1) * self.nfiber, iWave]
                    dx = xsample - xcen
                    dy = ysample - ycen
                    dd = np.zeros((len(xsample), 2))
                    dd[:, 0] = dx.flatten()
                    dd[:, 1] = dy.flatten()
                    self.flux[iExp * self.nfiber:(iExp + 1) * self.nfiber, iWave] = interpolate.interpn(
                        (self.xkernel, self.ykernel), self.kernelvalue[iWave, iExp], dd, method='linear',
                        bounds_error=False, fill_value=0.)
            else:
                xsample = self.xpos[:, iWave]
                ysample = self.ypos[:, iWave]
                dx = xsample - xcen
                dy = ysample - ycen
                dd = np.zeros((len(xsample), 2))
                dd[:, 0] = dx.flatten()
                dd[:, 1] = dy.flatten()
                self.flux[:, iWave] = interpolate.interpn((self.xkernel, self.ykernel), self.kernelvalue[iWave, 0], dd,
                                                          method='linear',
                                                          bounds_error=False, fill_value=0.)

    def set_flux_rss(self):
        """Set the flux to the RSS input values

        Notes:
        -----

        Only uses wavelengths specified by the object's waveindex attribute.

        Sets attributes:

         .flux - flux in each fiber [nExp * nfiber, nWave]
         .flux_ivar - inverse variance of flux in each fiber
                      [nExp * nfiber, nWave]
"""
        self.flux = self.rss.data['FLUX'].data
        self.flux_ivar = self.rss.data['IVAR'].data
        if (self.waveindex is not None):
            self.flux = self.flux[:, self.waveindex]
            self.flux_ivar = self.flux_ivar[:, self.waveindex]
        self.kernelvalue = np.zeros(self.nWave)

    def set_weights(self, **kwargs):
        """Set weights for fibers

        Notes:
        -----

        Uses the create_weights() method for the object. Loops over
        wavelength and stores the weights.

        Sets the .weights attribute to an ndarray of np.float32, of
        the shape: [nWave, nside * nside, nExp * nfiber]

"""
        self.weights = np.zeros([self.nWave, self.nside * self.nside,
                                 self.nExp * self.nfiber])
        for iWave in np.arange(self.nWave):
            wwT = self.create_weights(xsample=self.xpos[:, iWave],
                                      ysample=self.ypos[:, iWave],
                                      ivar=self.flux_ivar[:, iWave],
                                      kernel=self.kernelvalue[iWave],
                                      **kwargs)
            self.weights[iWave, :, :] = wwT

    def create_weights(self, xsample=None, ysample=None, ivar=None, kernel=None):
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

    def set_cube(self):
        """Set cube for each wavelength

        Notes:
        -----

        Sets the attributes:

          .cube - cube as ndarray of np.float32
                  [nside, nside, nWave]
          .cube_ivar - inverse variance of cube as ndarray of np.float32
                       [nside, nside, nWave]

        Full covariance matrix is accessible from the covar() method.
"""
        self.cube = np.zeros([self.nside, self.nside, self.nWave],
                             dtype=np.float32)
        self.cube_ivar = np.zeros([self.nside, self.nside, self.nWave],
                                  dtype=np.float32)
        for iWave in np.arange(self.nWave):
            fcube = ((self.weights[iWave].dot(self.flux[:, iWave])).reshape(self.nside,
                                                                            self.nside) *
                     self.conversion)
            self.cube[:, :, iWave] = fcube
            covar = self.covar(iWave)
            var = np.diagonal(covar)
            igt0 = np.where(var > 0)[0]
            ivar = np.zeros(self.nside * self.nside, dtype=np.float32)
            ivar[igt0] = 1. / var[igt0]
            self.cube_ivar[:, :, iWave] = ivar.reshape([self.nside,
                                                        self.nside])

    def covar(self, iWave=None):
        """Return cube covariance matrix for a wavelength

        Parameters:
        ----------

        iWave : int, np.int32
            index of wavelength to calculate for

        Returns:
        -------

        covar : ndarray of np.float32 [nside * nside, nside * nside]
"""
        iok = np.where(self.flux_ivar[:, iWave] > 0)[0]
        wwT = (self.weights[iWave, :, :])[:, iok]
        covar = wwT.dot(np.diag(self.flux_ivar[iok, iWave])).dot(wwT.T)
        return (covar)

    def plot_slice(self, iWave=None, vmax=None, vmin=0.):
        """Plot a slice of the cube"""
        map = self.cube[:, :, iWave]
        if (vmax is None):
            vmax = map.max() * 1.02
        extent = (self.xmin, self.xmax, self.ymin, self.ymax)
        plt.figure(figsize=(6.5, 5.))
        font = {'family': 'sans-serif',
                'size': 15}
        plt.rc('font', **font)
        plt.imshow(self.cube[:, :, iWave], extent=extent, vmin=vmin,
                   vmax=vmax, cmap=cm.gray_r, origin='lower')
        plt.xlabel('X (arcsec)')
        plt.ylabel('Y (arcsec)')
        plt.title('reconstruction of slice')
        plt.colorbar(label='flux')


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

    expindex : int, np.int32
        indices of exposures to use (default None)

    Methods:
    -------

    set_rss() : Acquire the RSS data

    set_flux_rss() : Set the flux used for reconstruction from RSS

    set_flux_psf() : Set the flux used for reconstruction to a PSF

    set_image_grid() : Set up the spatial grid for the cube

    psf(fwhm, x, y) : Return values of the PSF shape

    set_kernel() : Create the kernel estimation

    normalize_weights() : Normalize the weights

    create_weights() : Calculate the weights for mapping spectra to cube

    set_weights() : Set the weights for mapping spectra to cube

    set_cube() : Calculates and sets cube for all wavelengths

    covar() : Calculates covariance matrix for a slice of the cube

    plot_slice() : Plots a slice of the cube

    Notes:
    ------

    Additional attributes are set by the methods, as documented.

    Unless waveindex or expindex is set, uses all wavelengths and all
    exposures.

    Typical usage would be (for a limited number of wavelengths):

     import reconstruction
     r = reconstruction.Reconstruct(plate=8485, ifu=3701, waveindex=[1000, 2000])
     r.set_rss() # Gets and reads in the RSS file (sets .rss attribute)
     r.set_flux_rss() # Sets up to use the RSS fluxes (sets .flux, .flux_ivar)
     r.set_image_grid() # Creates the output image spatial grid
     r.set_weights() # Sets the weights (sets .weights)
     r.set_cube() # Sets the weights (sets .cube, .cube_ivar)
     r.plot_slice(0)

    To test performance on a PSF, you would want instead of set_flux_rss():

     r.set_kernel()  # Sets the kernel
     r.set_flux_psf(xcen=0.5, ycen=1.5)  # Puts fake data into fluxes

"""

    #     def __init__(self, plate=None, ifu=None, release='MPL-5',waveindex=None, expindex=None,keyword=None):
    #         Reconstruct.__init__(self, plate=plate, ifu=ifu, release=release,waveindex=waveindex, expindex=expindex)
    #         self.set_rss()
    #         self.set_image_grid()
    #         self.set_kernel()
    #         if keyword=='simulation':
    #             self.set_flux_psf()
    #         elif keyword == 'real':
    #             self.set_flux_rss()
    #         self.set_weights()
    #         self.set_cube()

    def create_weights(self, xsample=None, ysample=None,
                       ivar=None, kernel=None, shepard_sigma=0.7):
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

        w = np.exp(- 0.5 * dr ** 2 / shepard_sigma ** 2)
        ifit = np.where(dr > 1.6)
        w[ifit] = 0

        wwT = self.normalize_weights(w)
        return wwT


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

    expindex : int, np.int32
        indices of exposures to use (default None)

    Methods:
    -------

    set_rss() : Acquire the RSS data

    set_flux_rss() : Set the flux used for reconstruction from RSS

    set_flux_psf() : Set the flux used for reconstruction to a PSF

    set_image_grid() : Set up the spatial grid for the cube

    psf(fwhm, x, y) : Return values of the PSF shape

    set_kernel() : Create the kernel estimation

    create_weights() : Calculate the weights for mapping spectra to cube

    set_weights() : Set the weights for mapping spectra to cube

    set_cube() : Calculates and sets cube for all wavelengths

    covar() : Calculates covariance matrix for a slice of the cube

    plot_slice() : Plots a slice of the cube

    Notes:
    ------

    Additional attributes are set by the methods, as documented.

    Unless waveindex or expindex is set, uses all wavelengths and all
    exposures.

    Typical usage would be (for a limited number of wavelengths):

     import reconstruction
     r = reconstruction.Reconstruct(plate=8485, ifu=3701, waveindex=[1000, 2000])
     r.set_rss() # Gets and reads in the RSS file (sets .rss attribute)
     r.set_image_grid() # Creates the output image spatial grid
     r.set_flux_rss() # Sets up to use the RSS fluxes (sets .flux, .flux_ivar)
     r.set_kernel()  # Sets the kernel
     r.set_weights() # Sets the weights (sets .weights)
     r.set_cube() # Sets the weights (sets .cube, .cube_ivar)
     r.plot_slice(0)

    To test performance on a PSF, you would want instead of set_flux_rss():

     r.set_kernel()  # Sets the kernel
     r.set_flux_psf(xcen=0.5, ycen=1.5)  # Puts fake data into fluxes
"""

    #     def __init__(self, plate=None, ifu=None, release='MPL-5',waveindex=None, expindex=None,keyword=None):
    #         Reconstruct.__init__(self, plate=plate, ifu=ifu, release=release,waveindex=waveindex, expindex=expindex)
    #         self.set_rss()
    #         self.set_image_grid()
    #         self.set_kernel()
    #         if keyword=='simulation':
    #             self.set_flux_psf()
    #         elif keyword == 'real':
    #             self.set_flux_rss()
    #         self.set_weights()
    #         self.set_cube()

    def set_Amatrix(self, xsample=None, ysample=None, kernel=None, ratio=30):
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
        dd = np.zeros((nsample * self.nside * self.nside, 2))
        dd[:, 0] = dx.flatten()
        dd[:, 1] = dy.flatten()
        if (self.expindex is not None):
            Afull = interpolate.interpn((self.ykernel, self.xkernel), kernel[0], dd, method='linear',
                                        bounds_error=False,
                                        fill_value=0.).reshape(nsample, self.nimage)
        else:
            Afull = np.zeros([nsample, self.nside ** 2])
            for iExp in range(self.nExp):
                Afull[iExp * self.nfiber:(iExp + 1) * self.nfiber] = interpolate.interpn((self.ykernel, self.xkernel),
                                                                                         kernel[iExp], dd[
                                                                                                       iExp * self.nfiber * self.nimage:(
                                                                                                                                        iExp + 1) * self.nfiber * self.nimage],
                                                                                         method='linear',
                                                                                         bounds_error=False,
                                                                                         fill_value=0.).reshape(
                    self.nfiber, self.nimage)

        ineg = np.where(Afull < 0.)
        Afull[ineg] = 0.

        Asum = Afull.sum(axis=0)
        ifit = np.where(Asum > Asum.max() * ratio * 1.e-2)[0]

        A = Afull[:, ifit]
        return (ifit, A)

    def create_weights(self, xsample=None, ysample=None, ivar=None, kernel=None, lam=1E-4, ratio=30):
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

        wwT : ndarray of np.float32
            normalized weights [nside * nside, nExp * nfiber]

        Notes:
        -----

        This version uses Shepards method.
"""
        self.ifit, A = self.set_Amatrix(xsample, ysample, kernel, ratio)
        self.nfit = len(A[0])
        [U, D, VT] = np.linalg.svd(np.dot(np.diag(np.sqrt(ivar)), A), full_matrices=False)
        Dinv = 1 / D

        Q = (np.dot(np.dot(VT.transpose(), np.diag(D)), VT))
        sl = Q.sum(axis=1)
        Rl = (Q.T / sl.T).T

        A_plus = np.dot(np.dot(VT.T, np.diag(Dinv)), U.T)
        T = np.dot(np.dot(Rl, A_plus),np.diag(np.sqrt(ivar)))
        wwT = self.set_reshape(T)
        return wwT

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

#-------------------------------------------------------

def set_G(plate=None, ifu=None,release='MPL-5', waveindex =None, expindex = None,keyword=None):

    base = ReconstructG(plate=plate,ifu=ifu,release=release,waveindex=waveindex,expindex=expindex)
    base.set_rss()
    base.set_image_grid()
    if keyword=='real':
        base.set_flux_rss()
        base.set_kernel()
    elif keyword == 'simulation':
        base.set_kernel()
        base.set_flux_psf()
    base.set_weights()
    base.set_cube()
    return base

def set_Shepard(plate=None, ifu=None,release='MPL-5', waveindex =None, expindex = None,keyword=None):
    base = ReconstructShepard(plate=plate,ifu=ifu,release=release,waveindex=waveindex,expindex=expindex)
    base.set_rss()
    base.set_image_grid()
    if keyword=='real':
        base.set_flux_rss()
    elif keyword == 'simulation':
        base.set_kernel()
        base.set_flux_psf()
    base.set_weights()
    base.set_cube()
    return base
