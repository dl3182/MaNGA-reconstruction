# This file is to get the reconstruction result from least square


from BaseReconstruct import *
from BaseInfo import *
import numpy as np
from scipy.linalg import eigh
import scipy.interpolate as interpolate
import time


class ReconstructG(BaseReconstruct):
    def __init__(self, base=None, dimage=0.5, lam=1E-3, ratio=25):
        #     def __init__(self, rssfile=None, base=None,dimage=0.5 ,nkernel=201,lam=1E-3,alpha=1,beta=1,ratio=25,waveindex=None):

        if (base.dimage != dimage):
            print('Pixel size cannot match')
            dimage = base.dimage

        self.__dict__ = base.__dict__.copy()

        ivar = self.ivar.copy()
        ivar[np.where(ivar > 0)] = 1

        self.ratio = ratio
        self.lam = lam

        # For a single slice
        if self.single:
            (F, G, G_ivar, F_2, G_2) = self.G_core(self.waveindex, ivar=ivar)
            G_cov = []
            G_cov.append(self.G_cov_2)
            Indicator = self.Indicate.copy()

        # For the spectrum
        else:
            start_time = time.time()
            nWave = self.nWave
            Indicator, F, F_2, G, G_2, G_ivar= [np.zeros([nWave, self.nside, self.nside])
                                                                              for i in range(5)]
            G_cov = []

            for iWave in self.range:
                if (iWave % 500 == 0):
                    print('G wavelength channel', iWave)
                (F[iWave], G[iWave], G_ivar[iWave], F_2[iWave], G_2[iWave]) = self.G_core(iWave, ivar=ivar)
                G_cov.append(self.G_cov_2)
                Indicator[iWave] = self.Indicate.copy()

            stop_time = time.time()
            print("G interation Time = %.2f" % (stop_time - start_time))

        # result assignment
        self.IMGresult = G
        self.PSFresult = G_2
        self.cov = G_cov
        self.ivariance = G_ivar
        self.Indicator = Indicator

        self.F = F
        self.F2 = F_2

        if (len(self.range) == self.nWave) and (self.single == False):
            self.analysis()

    # main part for G's method
    def G_core(self, waveindex, ivar):
        if self.single:
            self.set_Amatrix(xsample=self.xpos[:, waveindex], ysample=self.ypos[:, waveindex], kernel=self.kernelvalue)
        else:
            self.set_Amatrix(xsample=self.xpos[:, waveindex], ysample=self.ypos[:, waveindex],
                             kernel=self.kernelvalue[waveindex])
        G_ivar = self.set_fit(ivar=ivar[:, waveindex], lam=self.lam)
        (F, G) = self.solve_FG(self.value[:, waveindex])
        if self.single:
            (F_2, G_2) = self.solve_FG(value=self.value_PSF)
        else:
            (F_2, G_2) = self.solve_FG(value=self.value_PSF[waveindex])
        return (F, G, G_ivar, F_2, G_2)

    def set_Amatrix(self, xsample, ysample, kernel):
        nsample = len(xsample)
        dx = np.outer(xsample, np.ones(self.nimage)) - np.outer(np.ones(nsample), self.ximage)
        dy = np.outer(ysample, np.ones(self.nimage)) - np.outer(np.ones(nsample), self.yimage)
        dd = np.zeros((nsample * self.nside * self.nside, 2))
        dd[:, 0] = dx.flatten()
        dd[:, 1] = dy.flatten()
        Afull = np.zeros([nsample, self.nside ** 2])
        for iExp in range(self.nExp):
            Afull[iExp * self.nFiber:(iExp + 1) * self.nFiber] = interpolate.interpn((self.ykernel, self.xkernel),
                                                                                     kernel[iExp], dd[
                                                                                                   iExp * self.nFiber * self.nimage:(
                                                                                                                                    iExp + 1) * self.nFiber * self.nimage],
                                                                                     method='linear',
                                                                                     bounds_error=False,
                                                                                     fill_value=0.).reshape(self.nFiber,
                                                                                                            self.nimage)
        ineg = np.where(Afull < 0.)
        Afull[ineg] = 0.

        Asum = Afull.sum(axis=0)
        self.ifit = np.where(Asum > Asum.max() * self.ratio * 1.e-2)[0]

        self.A = Afull[:, self.ifit]
        self.nfit = len(self.A[0])

        self.num_remove = self.nside * self.nside - self.nfit
        self.Indicate = np.ones(self.nside * self.nside)
        self.Indicate[self.ifit] = 0
        self.Indicate = self.Indicate.reshape(self.nside, self.nside)
        return

    def set_fit(self, ivar, lam):
        A_T = self.A.T
        self.Nmatrix = np.diag(ivar)

        H = np.identity(self.nfit)

        self.M0 = np.dot(np.dot(A_T, self.Nmatrix), self.A)
        M0inv = np.linalg.pinv(self.M0)
        self.M1 = self.M0 + lam ** 2 * H
        self.M = self.M0 + 2 * lam ** 2 * H + lam ** 4 * np.dot(np.dot(H, M0inv), H)

        eigvalue, eigvector = eigh(self.M, eigvals_only=False)
        Q = (eigvector.dot(np.sqrt(np.diag(np.abs(eigvalue))))).dot(eigvector.T)
        sl = Q.sum(axis=1)
        self.Rl = (Q.T / sl.T).T
        cl = 1 / (sl * sl)

        # G_cov is always diagonal, G_cov_2 is the one we evaluate.
        # self.G_cov = np.dot(np.dot(self.Rl, np.linalg.pinv(self.M)), np.transpose(self.Rl))
        self.G_cov_2 = np.dot(
            np.dot(self.Rl, np.dot(np.dot(np.linalg.pinv(self.M1), self.M0), np.linalg.pinv(self.M1))),
            np.transpose(self.Rl)) * self.conversion ** 2
        G_var = self.set_reshape(self.G_cov_2.diagonal())
        G_var[self.Indicate == 1] = 1E12
        G_ivar = 1 / (G_var)
        return (G_ivar)

    # solve the reconstruction
    def solve_FG(self, value):
        a = np.dot(np.linalg.pinv(self.M1), np.dot(np.transpose(self.A), np.dot(self.Nmatrix, value))) * self.conversion
        atilde = np.dot(self.Rl, a)
        F = self.set_reshape(a)
        G = self.set_reshape(atilde)
        return (F, G)

    # convert 1D to 2D squares
    def set_reshape(self, inp):
        result = np.zeros(self.nside ** 2)
        result[self.ifit] = inp
        result = result.reshape(self.nside, self.nside)
        return result

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