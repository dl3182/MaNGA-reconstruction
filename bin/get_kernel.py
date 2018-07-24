#!/usr/bin/env python

import os
import sys
import numpy as np
from scipy import signal as sign
from astropy.io import fits

path = os.getenv('RECONSTRUCTION_DIR')


class kernel(object):
    def __init__(self, length=None, dkernel=None):
        self.rough_length = length
        self.dkernel = dkernel

    def _create_grid(self, rough_length=None, d=None):
        nside = ((np.int32(np.ceil(rough_length / d)) // 2) * 2 + 1)
        length = nside * d
        self.xmin = -0.5 * length
        self.xmax = 0.5 * length
        self.ymin = -0.5 * length
        self.ymax = 0.5 * length
        xi = np.linspace(self.xmin + 0.5 * d, self.xmax - 0.5 * d, nside)
        yi = np.linspace(self.ymin + 0.5 * d, self.ymax - 0.5 * d, nside)
        x2i, y2i = np.meshgrid(xi, yi)
        return (nside, length, x2i, y2i, xi, yi)

    def psf(self, fwhm, x, y):
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

    def set_kernel(self):
        (self.nkernel, self.length, self.x2k, self.y2k, self.xkernel, self.ykernel) = self._create_grid(
            self.rough_length, self.dkernel)

        radius = np.sqrt(self.x2k ** 2 + self.y2k ** 2)
        fiber = np.zeros([self.nkernel, self.nkernel],
                         dtype=np.float32)
        ifiber = np.where(radius < 1.0)
        fiber[ifiber] = 1.
        self.fiber = fiber / fiber.sum()

        # Now convolve with PSF for each exposure and wavelength
        fwhm = np.arange(0.5, 2.5, 0.01)
        nfwhm = len(fwhm)
        self.kernelvalue = np.zeros([nfwhm, self.nkernel, self.nkernel], dtype=np.float32)
        for index, ifwhm in enumerate(fwhm):
            psf0 = self.psf(ifwhm, self.x2k, self.y2k)
            self.kernelvalue[index, :, :] = sign.fftconvolve(self.fiber, psf0, mode='same')
        return

    def create_set(self):
        try:
            os.remove('kernel_database.fits')
        except OSError:
            pass
        hdr = fits.Header()
        hdr['LENGTH'] = (self.length, 'Length of kernel grid(arcsec)')
        hdr['DKERNEL'] = (self.dkernel, 'spacing of kernel grid(pixel/arcsec)')
        hdu = fits.PrimaryHDU(data=self.kernelvalue, header=hdr)
        hdul = fits.HDUList([hdu])
        dbfile = os.path.join(path, 'python', 'data', 'kernel_database.fits')
        hdul.writeto(dbfile)

if __name__ == "__main__":
    ker = kernel(length=8, dkernel=0.05)
    ker.set_kernel()
    ker.create_set()
