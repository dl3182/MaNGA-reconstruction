
from astropy.io import fits


def write(rssfile, datafile, filename):
    data = filename + ".fits".format(filename=filename)
    rssfile.data.writeto(data, clobber=True)
    dataCUBE = fits.open(data)
    print('dataCube', dataCUBE.filename)

    card_GFWHM = fits.Card('GFWHM', datafile.GFWHM, 'Reconstructed FWHM in g-band (arcsec)')
    card_RFWHM = fits.Card('RFWHM', datafile.RFWHM, 'Reconstructed FWHM in r-band (arcsec)')
    card_IFWHM = fits.Card('IFWHM', datafile.IFWHM, 'Reconstructed FWHM in i-band (arcsec)')
    card_ZFWHM = fits.Card('ZFWHM', datafile.ZFWHM, 'Reconstructed FWHM in z-band (arcsec)')
    card_WCS_1 = fits.Card('CRPIX1', 12, 'Reference pixel (1-indexed)')
    card_WCS_2 = fits.Card('CRPIX2', 12, 'Reference pixel (1-indexed)')
    card_WCS_3 = fits.Card('CRVAL1', rssfile.data['FLUX'].header['IFURA'])
    card_WCS_4 = fits.Card('CRVAL2', rssfile.data['FLUX'].header['IFUDEC'])
    card_WCS_5 = fits.Card('CD1_1', -0.5 / 3600)
    card_WCS_6 = fits.Card('CD1_2', 0.5 / 3600)
    card_WCS_7 = fits.Card('CTYPE1', 'RA---TAN')
    card_WCS_8 = fits.Card('CTYPE2', 'DEC---TAN')
    card_WCS_9 = fits.Card('CUNIT1', 'deg')
    card_WCS_10 = fits.Card('CUNIT2', 'deg')
    card_WCS_list = [card_WCS_1, card_WCS_2, card_WCS_3, card_WCS_4, card_WCS_5, card_WCS_6, card_WCS_7, card_WCS_8,
                     card_WCS_9, card_WCS_10]

    # Primary
    dataCUBE[0].header['BUNIT'] = ('1E-17 erg/s/cm^2/Ang/spaxel', 'Specific intensity (per spaxel)')
    dataCUBE[0].header['MASKNAME'] = ('MANGA_DRP3PIXMASK', 'Bits in sdssMaskbits.par used by mask extension')
    # FLUX
    dataCUBE['FLUX'].data = datafile.IMGresult
    dataCUBE['FLUX'].header['BUNIT'] = dataCUBE['PRIMARY'].header['BUNIT']
    dataCUBE['FLUX'].header['MASKNAME'] = dataCUBE['PRIMARY'].header['MASKNAME']
    dataCUBE['FLUX'].header.insert('EBVGAL', card_GFWHM, after=True)
    dataCUBE['FLUX'].header.insert('GFWHM', card_RFWHM, after=True)
    dataCUBE['FLUX'].header.insert('RFWHM', card_IFWHM, after=True)
    dataCUBE['FLUX'].header.insert('IFWHM', card_ZFWHM, after=True)

    dataCUBE['FLUX'].header.rename_keyword('CTYPE1', 'CTYPE3')
    dataCUBE['FLUX'].header.rename_keyword('CRPIX1', 'CRPIX3')
    dataCUBE['FLUX'].header.rename_keyword('CRVAL1', 'CRVAL3')
    dataCUBE['FLUX'].header.rename_keyword('CD1_1', 'CD3_3')
    dataCUBE['FLUX'].header.rename_keyword('CUNIT1', 'CUNIT3')
    dataCUBE['FLUX'].header.insert('CUNIT3', ('CRPIX1', int((datafile.nside + 2) / 2), 'Reference pixel (1-indexed)'),
                                   after=True)
    dataCUBE['FLUX'].header.insert('CRPIX1', ('CRPIX2', int((datafile.nside + 2) / 2), 'Reference pixel (1-indexed)'),
                                   after=True)
    dataCUBE['FLUX'].header.insert('CRPIX2', ('CRVAL1', rssfile.data['FLUX'].header['IFURA']), after=True)
    dataCUBE['FLUX'].header.insert('CRVAL1', ('CRVAL2', rssfile.data['FLUX'].header['IFUDEC']), after=True)
    dataCUBE['FLUX'].header.insert('CRVAL2', ('CD1_1', -datafile.dimage / 3600), after=True)
    dataCUBE['FLUX'].header.insert('CD1_1', ('CD1_2', datafile.dimage / 3600), after=True)
    dataCUBE['FLUX'].header.insert('CD1_2', ('CTYPE1', 'RA---TAN'), after=True)
    dataCUBE['FLUX'].header.insert('CTYPE1', ('CTYPE2', 'DEC---TAN'), after=True)
    dataCUBE['FLUX'].header.insert('CTYPE2', ('CUNIT1', 'deg'), after=True)
    dataCUBE['FLUX'].header.insert('CUNIT1', ('CUNIT2', 'deg'), after=True)
    dataCUBE['FLUX'].header['HDUCLAS1'] = 'CUBE'
    # MASK
    dataCUBE['MASK'].data = datafile.Indicator
    # IVAR
    dataCUBE['IVAR'].data = datafile.ivariance
    # IMG/PSF
    index1 = dataCUBE['FLUX'].header.index('CRPIX1')
    index2 = dataCUBE['FLUX'].header.index('CUNIT2')
    GIMG = fits.ImageHDU(data=datafile.GIMG, name='GIMG', header=dataCUBE['FLUX'].header[index1:index2 + 1])
    RIMG = fits.ImageHDU(data=datafile.RIMG, name='RIMG', header=dataCUBE['FLUX'].header[index1:index2 + 1])
    IIMG = fits.ImageHDU(data=datafile.IIMG, name='IIMG', header=dataCUBE['FLUX'].header[index1:index2 + 1])
    ZIMG = fits.ImageHDU(data=datafile.ZIMG, name='ZIMG', header=dataCUBE['FLUX'].header[index1:index2 + 1])
    GPSF = fits.ImageHDU(data=datafile.GPSF, name='GPSF', header=dataCUBE['FLUX'].header[index1:index2 + 1])
    RPSF = fits.ImageHDU(data=datafile.RPSF, name='RPSF', header=dataCUBE['FLUX'].header[index1:index2 + 1])
    IPSF = fits.ImageHDU(data=datafile.IPSF, name='IPSF', header=dataCUBE['FLUX'].header[index1:index2 + 1])
    ZPSF = fits.ImageHDU(data=datafile.ZPSF, name='ZPSF', header=dataCUBE['FLUX'].header[index1:index2 + 1])
    #     RIMG.header.insert('GCOUNT',(dataCUBE['FLUX'].header[80:90]),after=True)
    dataCUBE.append(GIMG)
    dataCUBE.append(RIMG)
    dataCUBE.append(IIMG)
    dataCUBE.append(ZIMG)
    dataCUBE.append(GPSF)
    dataCUBE.append(RPSF)
    dataCUBE.append(IPSF)
    dataCUBE.append(ZPSF)

    PSF = fits.ImageHDU(data=datafile.PSFresult, name='PSF')
    dataCUBE.append(PSF)

    # F = fits.ImageHDU(data=datafile.F, name='F')
    # F2 = fits.ImageHDU(data=datafile.F2, name='F2')
    # dataCUBE.append(F)
    # dataCUBE.append(F2)

    del dataCUBE['DISP']
    del dataCUBE['XPOS']
    del dataCUBE['YPOS']

    dataCUBE.writeto(data, clobber=True)
    dataCUBE.close()
    return dataCUBE