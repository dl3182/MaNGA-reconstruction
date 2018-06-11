# This file is to get the reconstruction result from Shepard's method

from BaseReconstruct import *
from BaseInfo import *
import numpy as np
import time


class ReconstructShep(object):
    def __init__(self, base=None, dimage=0.5):
        #     def __init__(self, rssfile=None, dimage=0.5,nkernel=201,alpha=1,beta=1,waveindex=None):
        #         BaseReconstruct.__init__(self,rssfile=rssfile, dimage=dimage,nkernel=nkernel,alpha=1,beta=1,waveindex=waveindex)
        if (base.dimage != dimage):
            print('Pixel size cannot match')
        self.__dict__ = base.__dict__.copy()

        Shepard_cov = []
        if self.single:
            Shepard, Shepard_2, Shep_cov, Shepard_ivar, Shepard_cons = self.Shepard_core(self.waveindex)

            Shepard_cov.append(Shep_cov)
            Indicator = self.Indicate.copy()
        else:
            start_time = time.time()

            nWave = base.nWave
            Indicator, Shepard, Shepard_2, Shepard_ivar, Shepard_cons = (np.zeros((nWave, self.nside, self.nside)) for i
                                                                         in range(5))

            for iWave in self.range:
                if (iWave % 1000 == 0):
                    print('Shepard wavelength channel', iWave)
                Shepard[iWave], Shepard_2[iWave], Shep_cov, Shepard_ivar[iWave], Shepard_cons[
                    iWave] = self.Shepard_core(iWave)
                Shepard_cov.append(Shep_cov)
                Indicator[iWave] = self.Indicate.copy()

            stop_time = time.time()
            print("Shepard interation Time = %.2f" % (stop_time - start_time))

        self.IMGresult = Shepard
        self.PSFresult = Shepard_2
        self.cov = Shepard_cov
        self.ivariance = Shepard_ivar
        self.Indicator = Indicator

        self.PSF_cons = Shepard_cons

        if ((len(self.range) == self.nWave) and (self.waveindex == None)):
            self.analysis()

    def Shepard_core(self, iWave):
        Shep_cov, Shepard_ivar = self.set_fit(ivar=self.ivar[:, iWave], xsample=self.xpos[:, iWave],
                                              ysample=self.ypos[:, iWave])
        Shepard = self.solve_Shepard(value=self.value[:, iWave])
        if self.single:
            Shepard_2 = self.solve_Shepard(value=self.value_PSF)
            Shepard_cons = self.solve_Shepard(value=self.value_cons)
        else:
            Shepard_2 = self.solve_Shepard(value=self.value_PSF[iWave])
            Shepard_cons = self.solve_Shepard(value=self.value_cons[iWave])
        return (Shepard, Shepard_2, Shep_cov, Shepard_ivar, Shepard_cons)

    def set_fit(self, ivar, xsample, ysample, shepard_sigma=0.7):
        nsample = len(xsample)
        dx = np.outer(xsample, np.ones(self.nimage)) - np.outer(np.ones(nsample), self.ximage)
        dy = np.outer(ysample, np.ones(self.nimage)) - np.outer(np.ones(nsample), self.yimage)
        dr = np.sqrt(dx ** 2 + dy ** 2)
        w = np.transpose(np.matlib.repmat(ivar != 0, self.nside ** 2, 1))*np.exp(- 0.5 * dr ** 2 / shepard_sigma ** 2)
        ifit = np.where(dr > 1.6)
        w[ifit] = 0

        self.ifit = np.where(w.max(axis=0) > 0)[0]
        # normalize weight
        wsum = w.sum(axis=0)
        Indicate = np.ones(self.nside * self.nside)

        self.Indicate = Indicate.reshape(self.nside, self.nside)

        self.ww = np.zeros(w.shape)
        for i in range(self.nimage):
            if wsum[i] == 0:
                self.ww[:, i] = 0
            else:
                self.ww[:, i] = w[:, i] / wsum[i]
        self.wwT = self.ww.T
        Shep_cov = self.wwT.dot(self.ww) * self.conversion ** 2
        Shep_ivar = 1 / (np.diag(Shep_cov).reshape(self.nside, self.nside) + 1E-50 * np.ones([self.nside, self.nside]))
        Shep_ivar[np.where(Shep_ivar > 1E30)] == 0
        return (Shep_cov, Shep_ivar)

    def solve_Shepard(self, value):
        Shep = (self.wwT.dot(value)).reshape(self.nside, self.nside) * self.conversion
        return Shep

    def analysis(self):
        self.GPSF = PSFaverage('g', self.wave, self.PSFresult)
        self.GIMG = PSFaverage('g', self.wave, self.IMGresult)
        self.RPSF = PSFaverage('r', self.wave, self.PSFresult)
        self.RIMG = PSFaverage('r', self.wave, self.IMGresult)
        self.IPSF = PSFaverage('i', self.wave, self.PSFresult)
        self.IIMG = PSFaverage('i', self.wave, self.IMGresult)
        self.ZPSF = PSFaverage('z', self.wave, self.PSFresult)
        self.ZIMG = PSFaverage('z', self.wave, self.IMGresult)
        self.GFWHM = FWHM(self.xi, self.yi, self.GPSF)
        self.RFWHM = FWHM(self.xi, self.yi, self.RPSF)
        self.IFWHM = FWHM(self.xi, self.yi, self.IPSF)
        self.ZFWHM = FWHM(self.xi, self.yi, self.ZPSF)