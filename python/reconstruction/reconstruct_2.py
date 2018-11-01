import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate as interpolate
import marvin.tools.rss as rss
from marvin import config
from scipy.optimize import leastsq
from astropy.io import fits
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

    addexps : int, np int32, whether to add additional dithered exposures (default: None)
        addexps=1 means triple times exposures, addexps=2 means nine times exposures

    Methods:
    -------

    set_rss() : Acquire the RSS data

    set_image_grid(dimage) : Set up the spatial grid for the cube

    set_kernel(fwhm) : Create the kernel estimation from accessing to data base

    set_flux_rss() : Set the flux used for reconstruction from RSS

    set_flux_psf() : Set the flux used for reconstruction to a PSF

    set_weights() : Set the weights for mapping spectra to cube

    create_weights() : Calculate the weights for mapping spectra to cube

    normalize_weights(w) : Normalize the weights

    set_cube() : Calculates and sets cube for both RSS and simulation

    calculate_cube(self,flux,flux_ivar): Calculate the result for given flux and flux_ivar

    covar() : Calculates covariance matrix for a slice of the cube

    plot_slice() : Plots a slice of the cube

    Notes:
    ------

    Additional attributes are set by the methods, as documented.

    Unless waveindex is set, uses all wavelengths.

    Typical usage would be (for a limited number of wavelengths):

     import reconstruction
     r = reconstruction.Reconstruct(plate=8485, ifu=3701, waveindex=[1000, 2000],addexps=None)
     r.set_rss() # Gets and reads in the RSS file (sets .rss attribute)
     r.set_image_grid() # Creates the output image spatial grid
     r.set_kernel() # Sets the kernel for every wavelength and exposure
     r.set_flux_rss() # Sets up to use the RSS fluxes (sets .flux, .flux_ivar)
     r.set_flux_psf(xcen=0.5, ycen=1.5, alpha=1,noise=10) # Puts fake data into PSF fluxes (sets .flux_psf, .flux_psf_ivar)
     r.set_weights() # Sets the weights (sets .weights)
     r.set_cube() # Sets the weights (sets .cube, .cube_ivar)
     r.plot_slice(0)

"""

    def __init__(self, plate=None, ifu=None, release='MPL-5', waveindex=None, addexps=None):
        self.plate = plate
        self.ifu = ifu
        self.plateifu = "{plate}-{ifu}".format(plate=self.plate, ifu=self.ifu)
        self.release = release
        self.nfiber = int(self.ifu / 100)
        self.rss = None
        self.waveindex = waveindex
        self.addexps = addexps
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

        Notes:
        -----

        Sets attributes:

         .rss - Marvin RSS object
         .nExp - Number of exposures
         .xpos - X positions of each fiber [3**addexps * nExp * nfiber, nWave]
         .ypos - Y positions of each fiber [3**addexps * nExp * nfiber, nWave]
         .wave - Selected wavelength grid [nWave]
         .nWave - Size of wavelength grid
         .obsinfo - Observing information object
         .fwhm0 - FWHM of seeing at guider wavelength (5500 Angstroms) [nExp]
         .fwhm - FWHM at each wavelength [nWave, nExp]
"""
        self.rss = rss.RSS(plateifu=self.plateifu, release=self.release)
        self.nExp = self.rss.data[0].header['NEXP']
        self.xpos = self.rss.data['XPOS'].data
        self.ypos = self.rss.data['YPOS'].data
        if (self.addexps is None):
            self.addexps = 0

        if (self.addexps == 1) or (self.addexps == 2):
            xpos1 = self.rss.data['XPOS'].data + 2.5 / 3
            ypos1 = self.rss.data['YPOS'].data
            xpos2 = self.rss.data['XPOS'].data + 2.5 / 3 / 2
            ypos2 = self.rss.data['YPOS'].data - 2.5 / 6 * np.sqrt(3)
            self.xpos = np.concatenate((np.concatenate((self.xpos, xpos1), axis=0), xpos2), axis=0)
            self.ypos = np.concatenate((np.concatenate((self.ypos, ypos1), axis=0), ypos2), axis=0)
            if (self.addexps == 2):
                xpos1 = self.xpos + 2.5 / 6
                ypos1 = self.ypos + 2.5 / 6 / np.sqrt(3)
                xpos2 = self.xpos + 2.5 / 6
                ypos2 = self.ypos - 2.5 / 6 / np.sqrt(3)

                self.xpos = np.concatenate((np.concatenate((self.xpos, xpos1), axis=0), xpos2), axis=0)
                self.ypos = np.concatenate((np.concatenate((self.ypos, ypos1), axis=0), ypos2), axis=0)
        elif (self.addexps != 0):
            print("Only one and two additional exposure sets are provided")
            self.addexps = 0

        self.wave = self.rss.data['WAVE'].data
        self.nWave = self.rss.data['FLUX'].shape[1]
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
        index = int(fwhm * 100 - 50)
        index2 = fwhm * 100 - 50 - index
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

        noise : S/N ratio of simulated flux

        Notes:
        -----

        Requires set_kernel() to have already been called to set the
        flux for each exposure and wavelength.

        Only uses wavelengths specified by the object's waveindex attribute.

        Sets attributes:

         .flux_psf - flux in each fiber for simulation [3**addexps * nExp * nfiber, nWave]
         .flux_psf_ivar - inverse variance of flux in each fiber for simulation
                      [3**addexps * nExp * nfiber, nWave]
         .imagevalue - simulated image [nExp, nWave, nkernel, nkernel], only avaiable for 5 slices at most.

"""
        if (xcen is None):
            xcen = 0.
        if (ycen is None):
            ycen = 0.
        if (noise is None):
            noise = 0
        self.xcen = xcen
        self.ycen = ycen

        self.flux_psf = np.zeros([self.xpos.shape[0], self.nWave])
        self.flux_psf_ivar = np.ones([self.xpos.shape[0], self.nWave],
                                     dtype=np.float32)
        # start_time = time.time()
        imagevalue = []
        for iWave in np.arange(self.nWave):
            for iExp in np.arange(3 ** self.addexps * self.nExp):
                xsample = self.xpos[iExp * self.nfiber:(iExp + 1) * self.nfiber, iWave]
                ysample = self.ypos[iExp * self.nfiber:(iExp + 1) * self.nfiber, iWave]
                dx = xsample
                dy = ysample
                dd = np.zeros((len(xsample), 2))
                dd[:, 0] = dx.flatten()
                dd[:, 1] = dy.flatten()
                if (iExp < self.nExp):
                    kernelvalue0 = self._kernel(self.fwhm[iWave, iExp] * alpha) * (self.dimage / self.dkernel) ** 2
                else:
                    kernelvalue0 = self._kernel(np.mean(self.fwhm[iWave, :]) * alpha) * (
                                                                                            self.dimage / self.dkernel) ** 2

                if (xcen != 0) or (ycen != 0):
                    indx = np.where((np.round(self.xkernel, 4) - xcen) <= 0)[0][-1]
                    indy = np.where((np.round(self.ykernel, 4) - ycen) <= 0)[0][-1]
                    kernelvalue = np.zeros([self.nkernel, self.nkernel])
                    ncen = int((self.nkernel - 1) / 2)
                    for i in range(np.maximum(ncen - indx, 0), np.minimum(self.nkernel + ncen - indx, self.nkernel)):
                        for j in range(np.maximum(ncen - indy, 0),
                                       np.minimum(self.nkernel + ncen - indy, self.nkernel)):
                            kernelvalue[i + indx - ncen, j + indy - ncen] = kernelvalue0[i, j]
                else:
                    kernelvalue = kernelvalue0.copy()

                self.flux_psf[iExp * self.nfiber:(iExp + 1) * self.nfiber, iWave] = interpolate.interpn(
                    (self.xkernel, self.ykernel), kernelvalue, dd, method='linear',
                    bounds_error=False, fill_value=0.)
                if (self.nWave <= 5):
                    imagevalue.append(kernelvalue)
        if (self.nWave <= 5):
            self.imagevalue = np.array(imagevalue).reshape(3 ** self.addexps * self.nExp, self.nWave, self.nkernel,
                                                           self.nkernel)
        if (noise != 0):
            self.flux_psf0 = self.flux_psf.copy()
            for i in range(len(self.flux_psf)):
                self.flux_psf[i] = np.random.poisson(self.flux_psf[i] * 10 ** noise) / 10 ** noise

        # stop_time = time.time()

    # print("fiber flux time = %.2f" % (stop_time - start_time))

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

    def set_cube(self, **kwargs):
        """Set cube for each wavelength

        Notes:
        -----

        Sets the attributes:

          .cube - cube of RSS as ndarray of np.float32
                  [nside, nside, nWave]
          .cube_ivar - inverse variance of cube of RSS as ndarray of np.float32
                       [nside, nside, nWave]
          .cube_psf - cube of simulation as ndarray of np.float32
                  [nside, nside, nWave]
          .cube_ivar - inverse variance of cube of simulation as ndarray of np.float32
                       [nside, nside, nWave]
"""
        self.cube_psf, self.cube_psf_ivar = self.calculate_cube(self.flux_psf, self.flux_psf_ivar, **kwargs)
        self.cube, self.cube_ivar = self.calculate_cube(self.flux, self.flux_ivar, **kwargs)
        return

    def calculate_cube(self, flux, flux_ivar, **kwargs):
        """calculate cube and cube inverse variance for given flux

        Parameters:
        ----------

        flux: ndarray of np.float32
            flux of each fiber [nExp * nfiber]

        flux_ivar: ndarray of np.float32
            flux inverse variance of each fiber [nExp * nfiber]

        Return:
        ------

        cube : ndarray of np.float32 [nside, nside, nWave]
            reconstruction result
        cube_ivar : ndarray of np.float32 [nside, nside, nWave]
            inverse variance of reconstruction result


"""
        cube = np.zeros([self.nside, self.nside, self.nWave],
                        dtype=np.float32)
        cube_ivar = np.zeros([self.nside, self.nside, self.nWave],
                             dtype=np.float32)
        for iWave in np.arange(self.nWave):
            weights = self.create_weights(xsample=self.xpos[0:self.nExp * self.nfiber, iWave],
                                          ysample=self.ypos[0:self.nExp * self.nfiber, iWave],
                                          ivar=flux_ivar[:, iWave],
                                          waveindex=iWave, **kwargs)
            fcube = ((weights.dot(flux[:, iWave])).reshape(self.nside,
                                                           self.nside) *
                     self.conversion)
            cube[:, :, iWave] = fcube
            covar = self.covar(iWave, flux_ivar, weights)
            var = np.diagonal(covar)
            igt0 = np.where(var > 0)[0]
            ivar = np.zeros(self.nside * self.nside, dtype=np.float32)
            ivar[igt0] = 1. / var[igt0]
            cube_ivar[:, :, iWave] = ivar.reshape([self.nside,
                                                   self.nside])
        return (cube, cube_ivar)

    def covar(self, iWave=None, flux_ivar=None, weights=None):
        """Return cube covariance matrix for a wavelength

        Parameters:
        ----------

        iWave : int, np.int32
            index of wavelength to calculate for

        Returns:
        -------

        covar : ndarray of np.float32 [nside * nside, nside * nside]
"""
        iok = np.where(flux_ivar[:, iWave] > 0)[0]
        wwT = (weights[:, :])[:, iok]
        covar = wwT.dot(np.diag(flux_ivar[iok, iWave])).dot(wwT.T)
        return (covar)

    def plot_slice(self, iWave=0, keyword=None, vmax=None, vmin=0.):
        """Plot a slice of the cube"""
        if keyword == 'simulation':
            target = self.cube_psf[:, :, iWave]
        else:
            target = self.cube[:, :, iWave]
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

         .GPSF/RPSF/IPSF/ZPSF - ndarray of np.float32 [nside, nside] simulation image for each band
         .GIMG/RIMG/IIMG/ZIMG - ndarray of np.float32 [nside, nside] real image for each band
         .GFWHM/RFWHM/IFWHM/ZFWHM - float, np.float32,FWHM for each simulation image

"""
        self.GPSF = self.PSFaverage('g', self.wave, self.cube_psf)
        self.GIMG = self.PSFaverage('g', self.wave, self.cube)
        self.RPSF = self.PSFaverage('r', self.wave, self.cube_psf)
        self.RIMG = self.PSFaverage('r', self.wave, self.cube)
        self.IPSF = self.PSFaverage('i', self.wave, self.cube_psf)
        self.IIMG = self.PSFaverage('i', self.wave, self.cube)
        self.ZPSF = self.PSFaverage('z', self.wave, self.cube_psf)
        self.ZIMG = self.PSFaverage('z', self.wave, self.cube)
        self.GFWHM = self.FWHM(self.xi, self.yi, self.GPSF, xcen=self.xcen, ycen=self.ycen)[0]
        self.RFWHM = self.FWHM(self.xi, self.yi, self.RPSF, xcen=self.xcen, ycen=self.ycen)[0]
        self.IFWHM = self.FWHM(self.xi, self.yi, self.IPSF, xcen=self.xcen, ycen=self.ycen)[0]
        self.ZFWHM = self.FWHM(self.xi, self.yi, self.ZPSF, xcen=self.xcen, ycen=self.ycen)[0]

    def gaussian(self, rs, pars):
        """ gaussian function to fit the reconstruction image"""
        sig = pars[1]
        gaus = np.abs(pars[0]) * np.exp(-(rs * rs) / (2 * sig * sig)) / (2 * np.pi) / sig / sig
        return gaus

    def residuals(self, pars, rs, ks):
        """ the residuals of fitting for the reconstruction image"""
        err = ks - self.gaussian(rs, pars)
        return err

    def FWHM(self, xi=None, yi=None, PSF=None, xcen=0.0, ycen=0.0):
        """ calculate FWHM for given image

        Parameters:
        ----------

        xi : ndarray of np.float32
            X position of image grid

        yi : ndarray of np.float32
            Y position of image grid

        PSF : ndarray of np.float32 [nside,nside]
            image used to calculate FWHM

        xcen : float, np.float32
            X center of PSF desired

        ycen : float, np.float32
            Y center of PSF desired

        Returns:
        -------

        FWHM : float, np.float32,FWHM for the image

        Notes:
        -----

        This version uses gaussian function to fit the image
"""
        rng = 3.
        xs = xcen + 2. * rng * (np.random.random(size=1000) - 0.5)
        ys = ycen + 2. * rng * (np.random.random(size=1000) - 0.5)
        rs = np.sqrt((xs - xcen) ** 2 + (ys - ycen) ** 2)
        dx = xs
        dy = ys
        dd = np.zeros((len(xs), 2))
        dd[:, 0] = dy.flatten()
        dd[:, 1] = dx.flatten()
        ks = interpolate.interpn((yi, xi), PSF, dd, method='linear', bounds_error=False, fill_value=0.)

        pars = [1, 2]
        kernel_par = leastsq(self.residuals, pars, args=(rs, ks))

        rgrid = np.arange(100000) / 100000. * 5.
        imagewidth = rgrid[np.argmin(
            np.abs(self.gaussian(rgrid, kernel_par[0]).max() / 2 - self.gaussian(rgrid, kernel_par[0])))] * 2
        intensity = self.gaussian(rgrid[0], kernel_par[0])
        return (np.round(imagewidth, 3), intensity)

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
        PSF_ave = (np.matlib.repmat(band_value, n ** 2, 1) * (PSF.reshape(n ** 2, nWave))).reshape(n, n, nWave).sum(
            axis=2) / band_value.sum()
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

    addexps : int, np int32, whether to add additional dithered exposures (default: None)
        addexps=1 means triple times exposures, addexps=2 means nine times exposures

    Methods:
    -------

    set_rss() : Acquire the RSS data

    set_image_grid(dimage) : Set up the spatial grid for the cube

    set_kernel(fwhm) : Create the kernel estimation from accessing to data base

    set_flux_rss() : Set the flux used for reconstruction from RSS

    set_flux_psf() : Set the flux used for reconstruction to a PSF

    set_weights() : Set the weights for mapping spectra to cube

    create_weights() : Calculate the weights for mapping spectra to cube

    normalize_weights(w) : Normalize the weights

    set_cube() : Calculates and sets cube for both RSS and simulation

    calculate_cube(self,flux,flux_ivar): Calculate the result for given flux and flux_ivar

    covar() : Calculates covariance matrix for a slice of the cube

    plot_slice() : Plots a slice of the cube

    Notes:
    ------

    Additional attributes are set by the methods, as documented.

    Unless waveindex is set, uses all wavelengths.

    Typical usage would be (for a limited number of wavelengths):

     import reconstruction
     r = reconstruction.Reconstruct(plate=8485, ifu=3701, waveindex=[1000, 2000],addexps = None)
     r.set_rss() # Gets and reads in the RSS file (sets .rss attribute)
     r.set_image_grid() # Creates the output image spatial grid
     r.set_kernel() # Sets the kernel for every wavelength and exposure
     r.set_flux_rss() # Sets up to use the RSS fluxes (sets .flux, .flux_ivar)
     r.set_flux_psf(xcen=0.5, ycen=1.5, alpha=1, noise=10) # Puts fake data into PSF fluxes (sets .flux_psf, .flux_psf_ivar)
     r.set_weights() # Sets the weights (sets .weights)
     r.set_cube() # Sets the weights (sets .cube, .cube_ivar)
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

        w = np.transpose(np.matlib.repmat(ivar != 0, self.nside ** 2, 1)) * np.exp(- 0.5 * dr ** 2 / shepard_sigma ** 2)
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

    Methods:
    -------

    set_rss() : Acquire the RSS data

    set_image_grid(dimage) : Set up the spatial grid for the cube

    set_kernel(fwhm) : Create the kernel estimation from accessing to data base

    set_flux_rss() : Set the flux used for reconstruction from RSS

    set_flux_psf() : Set the flux used for reconstruction to a PSF

    set_weights() : Set the weights for mapping spectra to cube

    create_weights() : Calculate the weights for mapping spectra to cube

    normalize_weights(w) : Normalize the weights

    set_cube() : Calculates and sets cube for both RSS and simulation

    calculate_cube(self,flux,flux_ivar): Calculate the result for given flux and flux_ivar

    covar() : Calculates covariance matrix for a slice of the cube

    plot_slice() : Plots a slice of the cube

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
     r.set_flux_psf(xcen=0.5, ycen=1.5) # Puts fake data into PSF fluxes (sets .flux_psf, .flux_psf_ivar)
     r.set_weights() # Sets the weights (sets .weights)
     r.set_cube() # Sets the weights (sets .cube, .cube_ivar)
     r.plot_slice(0)

"""

    def set_Amatrix(self, xsample=None, ysample=None, waveindex=None, ratio=30, beta=1):
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
            kernel matrix [3**addexps * nExp * nfiber, nfit]

        Notes:
        -----

        indices will be used to recover A matrix back to regular grid
"""
        nsample = len(xsample)
        dx = np.outer(xsample, np.ones(self.nimage)) - np.outer(np.ones(nsample), self.x2i.flatten())
        dy = np.outer(ysample, np.ones(self.nimage)) - np.outer(np.ones(nsample), self.y2i.flatten())
        dr = np.sqrt(dx ** 2 + dy ** 2).flatten()
        dfwhm = (np.matlib.repmat(self.fwhm[waveindex], self.nfiber * self.nimage, 1).flatten('F'))*beta
        
        radius_lim=5.5
        indices=np.where(dr.flatten()<radius_lim)[0]

        dd = np.zeros([len(indices), 2])
        dd[:, 0] = dfwhm.flatten()[indices]
        dd[:, 1] = dr.flatten()[indices]
        
        ifwhm=np.arange(0.5,2.5,0.01)
        fwhmmin=int(dfwhm*100)-50
        fwhmmax=int(dfwhm*100)-50
        ifwhm=ifwhm[max(fwhmmin-3,0):min(fwhmmax+3,200)]
        
        ir=np.arange(0,radius_lim,0.05)
        
        Afull= interpolate.interpn((ifwhm, ir),self.kernel_radial[max(fwhmmin-3,0):min(fwhmmax+3,200),:], dd,method='linear',bounds_error=False,fill_value=0.)* (self.dimage / self.dkernel)**2
        Afull2 = np.zeros(len(dr.flatten()))
        Afull2[indices]=Afull
        Afull= Afull2.reshape(self.nExp*self.nfiber,self.nimage)

        ineg = np.where(Afull < 0.)
        Afull[ineg] = 0.

        Asum = Afull.sum(axis=0)
        ifit = np.where(Asum > Asum.max() * ratio * 1.e-2)[0]

        A = Afull[:, ifit]

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

        wwT : ndarray of np.float32
            normalized weights [nside * nside, nExp * nfiber]

        Notes:
        -----

        This version uses Shepards method.
"""
        self.ifit, A = self.set_Amatrix(xsample, ysample, waveindex, ratio, beta)
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


def set_base(plate=None, ifu=None, release='MPL-5', waveindex=None, addexps=None, dimage=0.75, dkernel=0.05, xcen=None,
             ycen=None, noise=None):
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
    base = Reconstruct(plate=plate, ifu=ifu, release=release, waveindex=waveindex, addexps=addexps)
    base.set_rss()
    base.set_image_grid(dimage=dimage)
    base.set_kernel(dkernel=dkernel)
    base.set_flux_rss()
    base.set_flux_psf(xcen=xcen, ycen=ycen, noise=noise)
    return base


def set_G(plate=None, ifu=None, release='MPL-5', waveindex=None, addexps=None, dimage=0.75, dkernel=0.05, lam=0,
          alpha=1, beta=1, xcen=None, ycen=None, noise=None):
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

        Return:
        --------
        base: object that includes information of grid and fiber fluxes

"""
    base = ReconstructG(plate=plate, ifu=ifu, release=release, waveindex=waveindex, addexps=addexps)
    base.set_rss()
    base.set_image_grid(dimage=dimage)
    base.set_kernel(dkernel=dkernel)
    base.set_flux_rss()
    base.set_flux_psf(alpha=alpha, xcen=xcen, ycen=ycen, noise=noise)
    start_time = time.time()
    base.set_cube(beta = beta, lam = lam)
    stop_time = time.time()
    print("calculation time = %.2f" % (stop_time - start_time))
    if (len(base.wave) == base.rss.data['FLUX'].data.shape[1]):
        base.set_band()
    return base


def set_Shepard(plate=None, ifu=None, release='MPL-5', waveindex=None, addexps=None, dimage=0.75, dkernel=0.05, alpha=1,
                beta=1, xcen=None, ycen=None, noise=None):
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

        Return:
        --------
        base: object that includes information of grid and fiber fluxes

"""
    base = ReconstructShepard(plate=plate, ifu=ifu, release=release, waveindex=waveindex, addexps=addexps)
    base.set_rss()
    base.set_image_grid(dimage=dimage)
    base.set_kernel(dkernel=dkernel)
    base.set_flux_rss()
    base.set_flux_psf(alpha=alpha, xcen=xcen, ycen=ycen, noise=noise)
    start_time = time.time()
    base.set_cube()
    stop_time = time.time()
    print("calculation time = %.2f" % (stop_time - start_time))
    if (len(base.wave) == base.rss.data['FLUX'].data.shape[1]):
        base.set_band()
    return base
