# BaseRun.basekernel(plate,ifu,dimage=0.75,nkernel=201,waveindex=None,add_exposures=None,single_kernel=None)

# parameters:
# plate,ifu: the plate-ifu of MaNGA data
# dimage: the pixel size of reconstruction
# nkernel: the number of pixels when constructing the kernel on a finer grid
# waveindex: the index of wavelength, None means the whole spectrum
# add_exposures: means whether add extra exposures for a simulation
#     0: actual dithering sampling
#     1: add the central point for ditherings
#     -1: dithering with only one set of observation
# single_kernel:means whether we use kernel for single exposure to speed up the program, True means using.
# return:
#     a class which includes the information of the kernel
# quantities:
#    measurements:nFiber,nExp,wave,nWave,xpos,ypos,nimage,nsample,value,ivar
#    regular grid: length,nside,waveindex,xi/yi
#    kernel:kernelvalue,imagevalue,value_PSF
# ---------------------------------------------------------------------------------------------------------------------
# BaseRun.Reconstruction(plate=None,ifu=None,dimage=0.75,nkernel=201,waveindex=None,ratio=25,lam=0,single_kernel=None):

# parameters:
# plate,ifu: the plate-ifu of MaNGA data
# dimage: the pixel size of reconstruction
# nkernel: the number of pixels when constructing the kernel on a finer grid
# waveindex: the index of wavelength, None means the whole spectrum
# ratio:the criterion to select the pixels in the G's configuration
# lam: the regularization coefficient for G's method
# single_kernel:means whether we use kernel for single exposure to speed up the program, True means using.
# cube: means whether we store the output into a fits file.

# return:
#    one class of grid and two classes which includes the information of Shepard and G
# quantities:
# IMGresult: the true reconstruction with regard to real flux
# PSFreulst: the simulation reconstruction with regard to a point source
# PSF_flat: the simulation result with regard to a flat field
# cov: the covariance of the reconstruction
# ivariance: the inverse of the diagonal of the covariance
# Indicator: the selected pixels for each wavelength
# For G, the quantities also include
#     F: the deconvolved result with regard to real flux
#     F2: the deconvolved result with regard to a point source
#     F_flat: the deconvolved result with regard to a flat field.


import time

from BaseRun import *

start_time = time.time()
plate=8551
ifu = 1902
sliceShepard,sliceG = Reconstruction(plate=plate,ifu=ifu,waveindex=2000)

base,Shepard,G = Reconstruction(plate=plate,ifu=ifu)

base=basekernel(plate=plate,ifu=ifu,add_exposures=1,single_kernel=10)

stop_time= time.time()

print("Time = %.2f"%(stop_time-start_time))
