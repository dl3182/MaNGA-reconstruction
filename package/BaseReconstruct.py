### This file is to generate kernel and simulated image for nExp


from BaseInfo import *
from ReconstructG import *
from ReconstructShep import *
import numpy as np
import math
from scipy import signal as sign
import scipy.interpolate as interpolate
import time

def Reconstruction(plate=None,ifu=None,dimage=0.75,dkernel=0.1,waveindex=None,ratio=25,lam=0,add_exposures=None,single_kernel=None,cube=None):
    rssfile=getrss(plate=plate,ifu=ifu)
    base=BaseReconstruct(rssfile=rssfile,dimage=dimage,dkernel=dkernel,waveindex=waveindex,add_exposures=add_exposures,single_kernel=single_kernel)
    Shepard=ReconstructShep(base=base,dimage=dimage)
    G=ReconstructG(base=base,dimage=dimage,ratio=ratio,lam=lam)
    if cube:
        Shepard_cube = writecube(rssfile, Shepard, 'Shepard-' + str(plate) + '-' + str(ifu) + '-LOGCUBE')
        G_cube = writecube(rssfile, G, 'G-' + str(plate) + '-' + str(ifu) + '-LOGCUBE')
    return (base,Shepard,G)


def basekernel(plate=None,ifu=None,dimage=0.75,dkernel=0.1,waveindex=None,add_exposures=None,single_kernel=None):
    rssfile=getrss(plate=plate,ifu=ifu)
    base=BaseReconstruct(rssfile=rssfile,dimage=dimage,dkernel=dkernel,add_exposures=add_exposures,single_kernel=single_kernel,waveindex=waveindex)
    return (base)


class BaseReconstruct(object):
    def __init__(self, rssfile=None, dimage=0.5, dkernel=0.1, alpha=1, beta=1, add_exposures=None, single_kernel=None,
                 waveindex=None):

        # reconstruction grid
        self.nFiber = int(rssfile.ifu / 100)
        self.waveindex = waveindex
        self.dimage = dimage

        self.conversion = dimage ** 2 / np.pi
        self.rough_length = getlength(self.nFiber)
        #         self.nside = np.int32(self.rough_length / self.dimage)
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
#         self.x2i, self.y2i = np.meshgrid(self.xi, self.yi)
        self.nimage = self.nside * self.nside

        # Information in RSS file
        self.nExp = rssfile.data[0].header['NEXP']
        self.xpos = rssfile.data['XPOS'].data
        self.ypos = rssfile.data['YPOS'].data
        self.wave = rssfile.data['WAVE'].data
        self.nWave = rssfile.data['FLUX'].shape[1]
        self.value = rssfile.data['FLUX'].data
        self.ivar = rssfile.data['IVAR'].data

        #### add exposures or not
        self.single_kernel = single_kernel
        if add_exposures:
            if (add_exposures == -1):
                self.xpos = rssfile.data['XPOS'].data[0 * 3 * self.nFiber:3 * self.nFiber]
                self.ypos = rssfile.data['YPOS'].data[0 * 3 * self.nFiber:3 * self.nFiber]
            else:
                xij = self.xpos.copy()
                yij = self.ypos.copy()
                self.xpos = np.zeros([1, xij.shape[1]])
                self.ypos = np.zeros([1, xij.shape[1]])
                for iset in range(int(self.nExp / 3)):
                    if (add_exposures == -2):
                        xiii = rssfile.data['XPOS'].data[(3 * iset + 1) * self.nFiber:(3 * iset + 2) * self.nFiber]
                        yiii = rssfile.data['YPOS'].data[(3 * iset + 1) * self.nFiber:(3 * iset + 2) * self.nFiber]
                    else:
                        # add points between neighbouring fibers
                        npoints = int(xij.shape[0] / self.nExp * 3)
                        xii = xij[iset * npoints:(iset + 1) * npoints]
                        yii = yij[iset * npoints:(iset + 1) * npoints]
                        di = np.zeros([xii.shape[0], xii.shape[0]])
                        for i in range(len(xii)):
                            for j in range(len(xii)):
                                di[i, j] = (xii[i, 100] - xii[j, 100]) ** 2 + (yii[i, 100] - yii[j, 100]) ** 2
                            di[i, i] = 100
                        near = np.zeros(di.shape)
                        for i in range(xii.shape[0]):
                            for j in np.arange(0, i):
                                if di[i, j] < 2.8:
                                    near[i, j] = 1
                        indx = np.where(near == 1)[0]
                        indy = np.where(near == 1)[1]

                        addx = np.zeros([len(indx), self.nWave])
                        addy = np.zeros([len(indx), self.nWave])
                        for i in range(len(indx)):
                            addx[i] = (xii[indx[i]] + xii[indy[i]]) / 2
                            addy[i] = (yii[indx[i]] + yii[indy[i]]) / 2

                        xiii = np.concatenate((xii, addx), axis=0)
                        yiii = np.concatenate((yii, addy), axis=0)
                    self.xpos = np.concatenate((self.xpos, xiii), axis=0)
                    self.ypos = np.concatenate((self.ypos, yiii), axis=0)
                self.xpos = self.xpos[1:]
                self.ypos = self.ypos[1:]

        self.nsample = self.xpos.shape[0]

        # kernel construction
        self.nkernel=np.int32(self.length/dkernel)
        self.xkernel = np.linspace(self.xmin + 0.5*dkernel, self.xmax - 0.5*dkernel,self.nkernel)
        self.ykernel = np.linspace(self.ymin + 0.5*dkernel, self.ymax - 0.5*dkernel,self.nkernel)
        self.x2k, self.y2k = np.meshgrid(self.xkernel, self.ykernel)
        
        # fiber
        self.radius = np.sqrt(self.x2k * self.x2k + self.y2k * self.y2k)
        self.fiber = np.zeros([self.nkernel, self.nkernel])
        ifiber = np.where(self.radius < 1.0)
        self.fiber[ifiber] = 1.
        self.fiber = self.fiber / self.fiber.sum()

        nWave = self.nWave
        self.range = np.arange(nWave)

        ## FWHM
        obsinfo = rssfile.data['OBSINFO'].data
        fwhm0 = obsinfo.field('SEEING') * obsinfo.field('PSFFAC')
        lambda0 = 5500
        self.fwhm = [[fwhm0[i] * math.pow(lambda0 / self.wave[j], 1 / 5) for i in range(fwhm0.shape[0])] for j in
                     range(nWave)]

        ## raw image
        signal = np.zeros((nWave, self.nside, self.nside))
        signal[:, int((self.nside - 1) / 2), int((self.nside - 1) / 2)] = 1

        if waveindex:
            self.single = True
            print('wavelength', self.wave[waveindex])
            kernelvalue, imagevalue, value_PSF = self.kernel_core(self.fwhm[waveindex], waveindex, signal[waveindex])
        else:
            self.single = False
            start_time = time.time()
            value_PSF = np.zeros([nWave, self.xpos.shape[0]])
            if self.single_kernel:
                kernelvalue = np.zeros([nWave, self.nkernel, self.nkernel])
                imagevalue = np.zeros([nWave, self.nside, self.nside])
            else:
                kernelvalue = np.zeros([nWave, self.nExp, self.nkernel, self.nkernel])
                imagevalue = np.zeros([nWave, self.nExp, self.nside, self.nside])

            for iWave in self.range:
                if (iWave % 1000 == 0):
                    print('kernel reconstruction', iWave)
                kernelvalue[iWave], imagevalue[iWave], value_PSF[iWave] = self.kernel_core(self.fwhm[iWave], iWave,
                                                                                           signal[iWave])
            stop_time = time.time()
            print("kernel construction time = %.2f" % (stop_time - start_time))

        # output of kernel reconstruction
        self.kernelvalue = kernelvalue
        self.value_PSF = value_PSF
        self.imagevalue = imagevalue
        self.value_flat = np.ones(self.value_PSF.shape)

    def kernel_core(self, fwhm, waveindex, signal):
        if self.single_kernel:
            kernelvalue, imagevalue = self.set_PSFvalue(fwhmw=fwhm[3], raw=signal)
            value_PSF = self.sample_value(self.xpos[:, waveindex], self.ypos[:, waveindex], kernelvalue)
        else:
            kernelvalue = np.zeros([self.nExp, self.nkernel, self.nkernel])
            imagevalue = np.zeros([self.nExp, self.nside, self.nside])
            value_PSF = np.zeros(self.xpos.shape[0])
            for iExp in range(self.nExp):
                kernelvalue[iExp], imagevalue[iExp] = self.set_PSFvalue(fwhmw=fwhm[iExp], raw=signal)
                value_PSF[iExp * self.nFiber:(iExp + 1) * self.nFiber] = self.sample_value(
                    xsample=self.xpos[iExp * self.nFiber:(iExp + 1) * self.nFiber, waveindex],
                    ysample=self.ypos[iExp * self.nFiber:(iExp + 1) * self.nFiber, waveindex],
                    kernelvalue=kernelvalue[iExp])
        return (kernelvalue, imagevalue, value_PSF)

    def PSFfcn(self, x, y, th, alpha):
        sig1 = th / 2.355 * alpha / 1.05
        self.th = sig1 * 2.355
        sig2 = 2. * sig1
        gaus1 = np.exp(-(x * x + y * y) / (2 * sig1 * sig1)) / (2 * np.pi) / sig1 / sig1
        gaus2 = np.exp(-(x * x + y * y) / (2 * sig2 * sig2)) / (2 * np.pi) / sig2 / sig2
        gaus = gaus1 + 1 / 9 * gaus2
        gaus = gaus / gaus.sum()
        return gaus

    # set kernel
    def set_PSFvalue(self, fwhmw, raw, alpha=1, beta=1):
        kernel = self.PSFfcn(self.x2k, self.y2k, fwhmw, alpha)
        kernel = sign.fftconvolve(self.fiber, kernel, mode='same')
        kernelvalue = kernel * self.nkernel ** 2 / self.nside ** 2

        dd = np.zeros((self.nimage, 2))
        dd[:, 0] = self.ximage
        dd[:, 1] = self.yimage
        PSFvalue = interpolate.interpn((self.ykernel, self.xkernel), kernelvalue, dd, method='linear',
                                       bounds_error=False, fill_value=0.).reshape(self.nside, self.nside)
        imagevalue = sign.convolve2d(raw, PSFvalue, mode='same', boundary='symm')
        return (kernelvalue, imagevalue)

    # set the fiber value
    def sample_value(self, xsample, ysample, kernelvalue, xpsf=0., ypsf=0.):
        dx = xsample - xpsf
        dy = ysample - ypsf
        dd = np.zeros((len(xsample), 2))
        dd[:, 0] = dx.flatten()
        dd[:, 1] = dy.flatten()
        value = interpolate.interpn((self.ykernel, self.xkernel), kernelvalue, dd, method='linear',
                                    bounds_error=False, fill_value=0.)
        return value
