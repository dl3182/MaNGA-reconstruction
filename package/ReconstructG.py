# This file is to get the reconstruction result from least square


from BaseReconstruct import *
from BaseInfo import *
import numpy as np
import scipy.interpolate as interpolate
import time


class ReconstructG(object):
    def __init__(self, base=None, dimage=0.5, lam=1E-3, ratio=25):
        #     def __init__(self, rssfile=None, base=None,dimage=0.5 ,dkernel=0.1,lam=1E-3,alpha=1,beta=1,ratio=25,waveindex=None):
        #         BaseReconstruct.__init__(self ,rssfile=rssfile, dimage=dimage ,dkernel=dkernel,alpha=1,beta=1,waveindex=waveindex)
        
        if (base.dimage != dimage):
            print('Pixel size cannot match')

        self.__dict__ = base.__dict__.copy()

        ivar = self.ivar.copy()
        ivar[np.where(ivar >= 0)] = 1

        self.ratio = ratio
        self.lam = lam

        if self.single:
            (F, G, G_ivar, F_2, G_2, F_flat, G_flat) = self.G_core(self.waveindex, ivar=ivar)
            G_cov = []
            G_cov.append(self.G_cov)
            Indicator = self.Indicate.copy()

        else:
            start_time = time.time()
            nWave = self.nWave
            Indicator, F, F_2, G, G_2, G_ivar, F_flat, G_flat = (np.zeros([nWave, self.nside, self.nside]) for i in
                                                                 range(8))
            G_cov = []

            for iWave in self.range:
                if (iWave % 500 == 0):
                    print('G wavelength channel', iWave)
                (F[iWave], G[iWave], G_ivar[iWave], F_2[iWave], G_2[iWave], F_flat[iWave], G_flat[iWave]) = self.G_core(
                    iWave, ivar=ivar)
                G_cov.append(self.G_cov)
                Indicator[iWave] = self.Indicate.copy()

            stop_time = time.time()
            print("G interation Time = %.2f" % (stop_time - start_time))

        ## G's output
        self.IMGresult = G
        self.PSFresult = G_2
        self.cov = G_cov
        self.ivariance = G_ivar
        self.Indicator = Indicator

        self.F = F
        self.F2 = F_2

        # flat field
        self.F_flat = F_flat
        self.PSF_flat = G_flat

        # calculate average for each band
        if (len(self.range) == self.nWave) and (self.single == False):
            self.analysis()

    def G_core(self, iWave, ivar):
        if self.single:
            self.set_Amatrix(xsample=self.xpos[:, iWave], ysample=self.ypos[:, iWave], kernel=self.kernelvalue)
        else:
            self.set_Amatrix(xsample=self.xpos[:, iWave], ysample=self.ypos[:, iWave], kernel=self.kernelvalue[iWave])
        G_ivar = self.set_fit(ivar=ivar[:, iWave], lam=self.lam)
        (F, G) = self.solve_FG(self.value[:, iWave])
        if self.single:
            (F_2, G_2) = self.solve_FG(value=self.value_PSF)
            (F_flat, G_flat) = self.solve_FG(self.value_flat)
        else:
            (F_2, G_2) = self.solve_FG(value=self.value_PSF[iWave])
            (F_flat, G_flat) = self.solve_FG(self.value_flat[iWave])
        return (F, G, G_ivar, F_2, G_2, F_flat, G_flat)

    # kernel matrix
    def set_Amatrix(self, xsample, ysample, kernel):
        dx = np.outer(xsample, np.ones(self.nimage)) - np.outer(np.ones(self.nsample), self.ximage)
        dy = np.outer(ysample, np.ones(self.nimage)) - np.outer(np.ones(self.nsample), self.yimage)
        dd = np.zeros((self.nsample * self.nside * self.nside, 2))
        dd[:, 0] = dx.flatten()
        dd[:, 1] = dy.flatten()
        if self.single_kernel:
            Afull = interpolate.interpn((self.ykernel, self.xkernel), kernel, dd, method='linear', bounds_error=False,
                                        fill_value=0.).reshape(self.nsample, self.nimage)
        else:
            Afull = np.zeros([self.nsample, self.nside ** 2])
            for iExp in range(self.nExp):
                Afull[iExp * self.nFiber:(iExp + 1) * self.nFiber] = interpolate.interpn((self.ykernel, self.xkernel),
                                                                                         kernel[iExp], dd[iExp * self.nFiber * self.nimage:(iExp + 1) * self.nFiber * self.nimage],
                                                                                         method='linear',
                                                                                         bounds_error=False,
                                                                                         fill_value=0.).reshape(self.nFiber, self.nimage)

        ineg = np.where(Afull < 0.)
        Afull[ineg] = 0.

        Asum = Afull.sum(axis=0)
        self.ifit = np.where(Asum > Asum.max() * self.ratio * 1.e-2)[0]

        self.A = Afull[:, self.ifit]
        self.nfit = len(self.A[0])

        self.Indicate = np.ones(self.nside * self.nside)
        self.Indicate[self.ifit] = 0
        self.Indicate = self.Indicate.reshape(self.nside, self.nside)
        return

    # transformation from fiber to regular grid with SVD
    def set_fit(self, ivar, lam):
        self.Nmatrix = np.diag(ivar)

        [U, D, VT] = np.linalg.svd(np.dot(np.diag(np.sqrt(ivar)), self.A), full_matrices=False)
        Dinv = 1 / D

        Q = (np.dot(np.dot(VT.transpose(), np.diag(D)), VT))
        sl = Q.sum(axis=1)
        self.Rl = (Q.T / sl.T).T

        self.A_plus = np.dot(np.dot(VT.T, np.diag(Dinv)), U.T)
        self.F_cov = np.dot(self.A_plus, self.A_plus.T)

        self.T = np.dot(self.Rl, self.A_plus)
        self.G_cov = np.dot(self.T, np.transpose(self.T)) * self.conversion ** 2
        G_var = self.set_reshape(self.G_cov.diagonal())
        G_var[self.Indicate == 1] = 1E12 # avoid 1/0
        G_ivar = 1 / (G_var)
        return G_ivar

    # result of reconstruction
    def solve_FG(self, value):
        a = np.dot(np.dot(self.A_plus, np.diag(np.sqrt(np.diag(self.Nmatrix)))), value) * self.conversion
        atilde = np.dot(self.Rl, a)
        F = self.set_reshape(a)
        G = self.set_reshape(atilde)
        return (F, G)

    # convert 1D result to 2D regular grid
    def set_reshape(self, inp):
        result = np.zeros(self.nside ** 2)
        result[self.ifit] = inp
        result = result.reshape(self.nside, self.nside)
        return result

    # result of each band
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
