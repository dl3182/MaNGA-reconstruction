### This file is to generate kernel and simulated image for nExp


from BaseInfo import *
import numpy as np
import math
from scipy import signal as sign
import scipy.interpolate as interpolate
import time


class BaseReconstruct(object):
    def __init__(self, rssfile=None, dimage=0.5, nkernel=201, alpha=1, beta=1, waveindex=None):
        self.nFiber = int(rssfile.ifu / 100)

        self.nkernel = nkernel
        self.dimage = dimage
        self.conversion = dimage ** 2 / np.pi
        self.rough_length = getlength(self.nFiber)
        # self.nside = np.int32(self.rough_length / self.dimage)
        self.nside = (np.int32(np.ceil(self.rough_length / self.dimage)) // 2) * 2 + 1
        self.length = self.nside * self.dimage
        self.xmin = -0.5 * self.length
        self.xmax = 0.5 * self.length
        self.ymin = -0.5 * self.length
        self.ymax = 0.5 * self.length
        self.ximage = (self.length * np.outer(np.ones(self.nside),
                                              (np.arange(self.nside) + 0.5) /
                                              np.float64(self.nside) - 0.5)).flatten()
        self.yimage = (self.length * np.outer((np.arange(self.nside) + 0.5) /
                                              np.float64(self.nside) - 0.5,
                                              np.ones(self.nside))).flatten()
        self.xi = np.linspace(self.xmin + 0.5 * self.dimage, self.xmax - 0.5 * self.dimage, self.nside)
        self.yi = np.linspace(self.ymin + 0.5 * self.dimage, self.ymax - 0.5 * self.dimage, self.nside)
        self.x2i, self.y2i = np.meshgrid(self.xi, self.yi)
        self.nimage = self.nside * self.nside

        self.nExp = rssfile.data[0].header['NEXP']
        self.xpos = rssfile.data['XPOS'].data
        self.ypos = rssfile.data['YPOS'].data
        self.value = rssfile.data['FLUX'].data
        self.ivar = rssfile.data['IVAR'].data
        self.nWave = rssfile.data['FLUX'].shape[1]
        self.wave = rssfile.data['WAVE'].data

        nWave = self.nWave
        self.range = np.arange(nWave)

        ## FWHM
        obsinfo = rssfile.data['OBSINFO'].data
        fwhm0 = obsinfo.field('SEEING') * obsinfo.field('PSFFAC')
        lambda0 = 5500

        self.fwhm = [[fwhm0[i] * math.pow(lambda0 / self.wave[j], 1 / 5) for i in range(fwhm0.shape[0])] for j in
                     range(nWave)]

        signal = np.zeros((nWave, self.nside, self.nside))
        signal[:, int((self.nside - 1) / 2), int((self.nside - 1) / 2)] = 1
        self.waveindex = waveindex

        if waveindex:
            self.single = True
            print('wavelength', self.wave[waveindex])
            kernelvalue, imagevalue, value_PSF = self.kernel_core(self.fwhm[waveindex], waveindex, signal[waveindex])
        else:
            self.single = False
            start_time = time.time()
            value_PSF = np.zeros([nWave, self.xpos.shape[0]])
            kernelvalue = np.zeros([nWave, self.nExp, self.nkernel, self.nkernel])
            imagevalue = np.zeros([nWave, self.nExp, self.nside, self.nside])

            for iWave in self.range:
                if (iWave % 500 == 0):
                    print('kernel reconstruction', iWave)
                kernelvalue[iWave], imagevalue[iWave], value_PSF[iWave] = self.kernel_core(self.fwhm[iWave], iWave,
                                                                                           signal[iWave])
            stop_time = time.time()
            print("kernel construction time = %.2f" % (stop_time - start_time))
        self.kernelvalue = kernelvalue;
        self.value_PSF = value_PSF
        self.imagevalue = imagevalue

    def kernel_core(self, fwhm, waveindex, signal):
        #         kernelvalue,imagevalue = self.set_PSFvalue(fwhmw=fwhm,raw=signal)
        #         value_PSF = self.sample_value(self.xpos[:,waveindex],self.ypos[:,waveindex],imagevalue)
        kernelvalue = np.zeros([self.nExp, self.nkernel, self.nkernel])
        imagevalue = np.zeros([self.nExp, self.nside, self.nside])
        value_PSF = np.zeros(self.xpos.shape[0])
        for iExp in range(self.nExp):
            kernelvalue[iExp], imagevalue[iExp] = self.set_PSFvalue(fwhmw=fwhm[iExp], raw=signal)
            value_PSF[iExp * self.nFiber:(iExp + 1) * self.nFiber] = self.sample_value(
                xsample=self.xpos[iExp * self.nFiber:(iExp + 1) * self.nFiber, waveindex],
                ysample=self.ypos[iExp * self.nFiber:(iExp + 1) * self.nFiber, waveindex], imagevalue=imagevalue[iExp])
        return (kernelvalue, imagevalue, value_PSF)

    def PSFfcn(self, x, y, th, alpha):
        sig1 = th / 2.355 * alpha / 1.05
        self.th = sig1 * 2.355
        sig2 = 2. * sig1
        gaus1 = np.exp(-(x * x + y * y) / (2 * sig1 * sig1)) / (2 * np.pi) / sig1 / sig1
        gaus2 = np.exp(-(x * x + y * y) / (2 * sig2 * sig2)) / (2 * np.pi) / sig2 / sig2
        gaus = gaus1 + 1 / 9 * gaus2
        return gaus

    def PSFfcn_2(self, x, y, th, beta):
        sig1 = th / 2.355 * beta / 1.05
        self.th = sig1 * 2.355
        sig2 = 2. * sig1
        gaus1 = np.exp(-(x * x + y * y) / (2 * sig1 * sig1)) / (2 * np.pi) / sig1 / sig1
        gaus2 = np.exp(-(x * x + y * y) / (2 * sig2 * sig2)) / (2 * np.pi) / sig2 / sig2
        gaus = gaus1 + 1 / 9 * gaus2
        return gaus

    def set_PSFvalue(self, fwhmw, raw, alpha=1, beta=1, r=1.0):
        n = self.nkernel
        self.xkernel = np.linspace(self.xmin + self.length / n, self.xmax - self.length / n, n)
        self.ykernel = np.linspace(self.ymin + self.length / n, self.ymax - self.length / n, n)
        x2k, y2k = np.meshgrid(self.xkernel, self.ykernel)
        radius = np.sqrt(x2k * x2k + y2k * y2k)
        fiber = np.zeros([n, n])
        ifiber = np.where(radius < r)
        fiber[ifiber] = 1.

        kernel = self.PSFfcn(x2k, y2k, fwhmw, alpha)
        kernel = sign.fftconvolve(fiber, kernel, mode='same')
        kernelvalue = kernel / kernel.sum() * n ** 2 / self.nside ** 2

        dd = np.zeros((self.nimage, 2))
        dd[:, 0] = self.x2i.flatten()
        dd[:, 1] = self.y2i.flatten()
        PSFvalue = interpolate.interpn((self.ykernel, self.xkernel), kernelvalue, dd, method='linear',
                                       bounds_error=False, fill_value=0.).reshape(self.nside, self.nside)
        imagevalue = sign.convolve2d(raw, PSFvalue, mode='same', boundary='symm')
        return (kernelvalue, imagevalue)

    def sample_value(self, xsample, ysample, imagevalue, xpsf=0., ypsf=0.):
        nsample = len(xsample)
        dx = xsample - xpsf
        dy = ysample - ypsf
        dd = np.zeros((nsample, 2))
        dd[:, 0] = dx.flatten()
        dd[:, 1] = dy.flatten()
        value = interpolate.interpn((self.yi, self.xi), imagevalue, dd, method='linear',
                                    bounds_error=False, fill_value=0.)
        return value