# This file is the base functions involved in the analysis

## functions will be used during the analysis
#-------------------------------------------------
# getrss(plate=None, ifu=None, release='MPL-5'):
# getcube(plate=None, ifu=None, release='MPL-5'):
#    input: plate,ifu
#    output: rss file
#-------------------------------------------------
# getlength(x):
# parameters:
#    x: number of fibers of ifu
# return:
#    the initial rough length of reconstruction grid in arcsec.
#-------------------------------------------------
# PSFaverage(color,wave,PSF):
# parameters:
#   color: the band of filter, which can choose,'g','r','i','z'
#   wave: the wavelength of the spectrum
#   PSF: the input image for the whole spectrum
#-------------------------------------------------
# gaussian(rs, pars):
# residuals(pars, rs, ks):
# FWHM(xi, yi, PSF, xpsf=0, ypsf=0):
# parameters:
#   xi,yi: coordinates of each pixel
#   PSF: the image used to calculate the FWHM
#   xpsf,ypsf: the offset of center
# return:
#   
#-------------------------------------------------

import numpy as np
import marvin.tools.rss as rss
from marvin.tools.cube import Cube
import scipy.interpolate as interpolate
from scipy.optimize import leastsq

def getrss(plate=None, ifu=None, release='MPL-5'):
    plateifu = "{plate}-{ifu}".format(plate=plate, ifu=ifu)
    r = rss.RSS(plateifu=plateifu, release=release)
    r.download()
    return (r)

def getcube(plate=None, ifu=None, release='MPL-5'):
    plateifu = "{plate}-{ifu}".format(plate=plate, ifu=ifu)
    r = Cube(plateifu=plateifu, release=release)
    r.download()
    return (r)

def getlength(x):
    return {
        7: 12,
        19: 17,
        37: 22,
        61: 27,
        91: 32,
        127: 36,
    }.get(x, None)


def PSFaverage(color, wave, PSF):
    band0 = np.loadtxt('../data/' + color + '_filter.dat')
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


def gaussian(rs, pars):
    sig = pars[1]
    gaus = np.abs(pars[0]) * np.exp(-(rs * rs) / (2 * sig * sig)) / (2 * np.pi) / sig / sig
    return gaus


def residuals(pars, rs, ks):
    err = ks - gaussian(rs, pars)
    return err


def FWHM(xi, yi, PSF, xpsf=0, ypsf=0):
    rng = 3.
    xs = xpsf + 2. * rng * (np.random.random(size=10000) - 0.5)
    ys = ypsf + 2. * rng * (np.random.random(size=10000) - 0.5)
    rs = np.sqrt((xs - xpsf) ** 2 + (ys - ypsf) ** 2)
    dx = xs
    dy = ys
    dd = np.zeros((len(xs), 2))
    dd[:, 0] = dx.flatten()
    dd[:, 1] = dy.flatten()
    ks = interpolate.interpn((yi, xi), PSF, dd, method='linear', bounds_error=False, fill_value=0.)

    pars = [1, 2]
    kernel_par = leastsq(residuals, pars, args=(rs, ks))

    rgrid = np.arange(100000) / 100000. * 5.
    imagewidth = rgrid[np.argmin(np.abs(gaussian(rgrid, kernel_par[0]).max() / 2 - gaussian(rgrid, kernel_par[0])))] * 2
    return imagewidth
